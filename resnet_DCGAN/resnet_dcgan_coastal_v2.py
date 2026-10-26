"""
ResNet-DCGAN 海岸線時序預測
用法:
  # (A) 先做留出驗證: 用 2013-2021 訓練, 預測 2022-2024 並評分
  python resnet_dcgan_coastal_v2.py --data_dir DIR --out_dir ./val --holdout 3 --epochs 150
"""
import argparse
import json
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------
# 1. 資料集
# ----------------------------------------------------------------------------
class CoastalSequenceDataset(Dataset):
    def __init__(self, data_dir, k=3, patch_w=384, patch_h=187,
                 samples_per_epoch=800, flip=False, holdout=0):
        paths = []
        for ext in ("jpg", "jpeg", "png", "tif", "tiff"):
            paths += glob(os.path.join(data_dir, f"*.{ext}"))
        year_path = {}
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                year_path[int(name)] = p
            except ValueError:
                continue
        years = sorted(year_path)
        assert len(years) >= k + 1 + holdout, "影像張數不足"
        gaps = [b - a for a, b in zip(years[:-1], years[1:]) if b - a != 1]
        if gaps:
            print(f"[warn] 年份不連續: {years}")

        self.imgs = {}
        for y in years:
            arr = np.asarray(Image.open(year_path[y]).convert("RGB")).astype(np.float32)
            self.imgs[y] = arr / 127.5 - 1.0
        shapes = {v.shape for v in self.imgs.values()}
        assert len(shapes) == 1, f"影像尺寸不一致: {shapes}"

        self.years = years
        self.h, self.w, _ = self.imgs[years[0]].shape
        self.k = k
        self.patch_w = min(patch_w, self.w)
        self.patch_h = min(patch_h, self.h)
        self.samples_per_epoch = samples_per_epoch
        self.flip = flip

        # 訓練可用的最後一年 (留出驗證時, 之後的年份完全不參與訓練)
        self.train_end = years[-1 - holdout]
        self.valid_starts = [i for i in range(len(years) - k)
                             if years[i + k] <= self.train_end]
        assert self.valid_starts, "訓練序列為空, 請減小 k 或 holdout"
        print(f"[data] years={years[0]}-{years[-1]}, train_end={self.train_end}, "
              f"train pairs={len(self.valid_starts)}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        i = random.choice(self.valid_starts)
        seq = self.years[i: i + self.k + 1]
        x = random.randint(0, self.w - self.patch_w)
        y = random.randint(0, self.h - self.patch_h) if self.h > self.patch_h else 0
        do_flip = self.flip and random.random() < 0.5

        def patch(year):
            a = self.imgs[year][y:y + self.patch_h, x:x + self.patch_w, :]
            return a[:, ::-1, :].copy() if do_flip else a

        past = np.concatenate([patch(s) for s in seq[:-1]], axis=-1)
        tgt = patch(seq[-1])
        return (torch.from_numpy(past).permute(2, 0, 1).float(),
                torch.from_numpy(tgt).permute(2, 0, 1).float())


# ----------------------------------------------------------------------------
# 2. 模型
# ----------------------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=3, ngf=64, n_resblocks=6, residual=False):
        """residual=True: 輸出 = 上一年影像 + 學到的變化量 (最後一層零初始化,
        一開始等於「直接複製上一年」, 靜態地物預設保留, 模型只需學會改變的部分)"""
        super().__init__()
        self.residual = residual
        self.register_buffer("residual_flag", torch.tensor(1 if residual else 0))
        m = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, ngf, 7),
             nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):
            m += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        m += [ResnetBlock(ch) for _ in range(n_resblocks)]
        for _ in range(2):
            m += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1),
                  nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        m += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_channels, 7)]
        if not residual:
            m += [nn.Tanh()]
        self.model = nn.Sequential(*m)
        if residual:
            nn.init.zeros_(self.model[-1].weight)
            nn.init.zeros_(self.model[-1].bias)

    def forward(self, x):
        # 內部會 2 次下採樣再上採樣, 尺寸需為 4 的倍數; 不是的話先補邊、輸出再裁回原尺寸
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        xin = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        out = self.model(xin)[..., :h, :w]
        if self.residual:
            out = (x[:, -3:] + out).clamp(-1, 1)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, n_layers=3):
        super().__init__()
        L = [nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers):
            s = 2 if i < n_layers - 1 else 1
            L += [nn.Conv2d(ch, ch * 2, 4, s, 1), nn.InstanceNorm2d(ch * 2), nn.LeakyReLU(0.2, True)]
            ch *= 2
        L += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.model = nn.Sequential(*L)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------------
# 3. 訓練
# ----------------------------------------------------------------------------
def train(args, device):
    ds = CoastalSequenceDataset(args.data_dir, k=args.k, patch_w=args.patch_w,
                                patch_h=args.patch_h,
                                samples_per_epoch=args.samples_per_epoch,
                                flip=args.flip, holdout=args.holdout)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    G = ResnetGenerator(3 * args.k, 3, args.ngf, args.n_resblocks, residual=args.residual).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    use_gan = args.adv_weight > 0
    if use_gan:
        D = PatchDiscriminator(3 * args.k + 3, args.ndf).to(device)
        opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1, mse = nn.L1Loss(), nn.MSELoss()
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        for past, target in loader:
            past, target = past.to(device), target.to(device)
            loss_D = torch.zeros(())
            if use_gan:
                with torch.no_grad():
                    fake = G(past)
                pr = D(torch.cat([past, target], 1))
                pf = D(torch.cat([past, fake], 1))
                loss_D = 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            fake = G(past)
            loss_l1 = l1(fake, target) * args.l1_weight
            loss_adv = torch.zeros((), device=device)
            if use_gan:
                pf = D(torch.cat([past, fake], 1))
                loss_adv = mse(pf, torch.ones_like(pf)) * args.adv_weight
            opt_G.zero_grad(); (loss_adv + loss_l1).backward(); opt_G.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[epoch {epoch:4d}/{args.epochs}] D={loss_D.item():.4f} "
                  f"G_adv={loss_adv.item():.4f} G_l1={loss_l1.item():.4f}")
    torch.save(G.state_dict(), os.path.join(args.out_dir, "generator.pt"))
    return G, ds


# ----------------------------------------------------------------------------
# 4. 集成式自回歸預測
# ----------------------------------------------------------------------------
def _pad4(arr):
    h, w, _ = arr.shape
    return np.pad(arr, ((0, (4 - h % 4) % 4), (0, (4 - w % 4) % 4), (0, 0)), mode="reflect")


def _perturb(x, noise_std, max_shift, rng):
    """G 是確定性的, 以輸入雜訊 + 小幅平移製造樣本間差異。
    回傳 (擾動後的輸入, 每個樣本的平移量); 輸出必須用 _unshift 平移回來,
    否則各樣本的輸出彼此錯位, 取中位數/平均會把影像磨糊, 且錯位會逐年累積。"""
    shifts = []
    for j in range(x.shape[0]):
        if max_shift > 0:
            dy, dx = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
            x[j] = torch.roll(x[j], shifts=(dy, dx), dims=(1, 2))
        else:
            dy = dx = 0
        shifts.append((dy, dx))
    x_pre = x.clone()
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x.clamp(-1, 1), shifts, x_pre


def _unshift(y, shifts):
    for j, (dy, dx) in enumerate(shifts):
        if dy or dx:
            y[j] = torch.roll(y[j], shifts=(-dy, -dx), dims=(1, 2))
    return y


@torch.no_grad()
def rollout(G, init_frames, n_future, n_samples, mode, noise_std, max_shift,
            batch, agg, device, seed=0, band_px=0, band_feather=8, bands=None, protect=None):
    """回傳 preds[list of HxWx3 in [-1,1]], stds[list of HxW]
    band_px>0: 只允許岸線附近 band_px 像素內被模型改動, 其餘保留上一年影像
               (解決逐年變糊與突堤/設施消失); bands 若給 list 會存入每年的權重圖。
    protect: HxW 權重圖 (見 coastal_metrics.make_protect), 1 的區域強制沿用上一年影像,
             即使落在岸線帶內也不讓模型改動 (用於保護細長的突堤、棧橋等容易消失的結構)。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    G.eval()
    if n_samples == 1:
        noise_std, max_shift = 0.0, 0
    h0, w0 = init_frames[0].shape[:2]
    base = torch.from_numpy(np.concatenate([_pad4(f) for f in init_frames], -1))
    base = base.permute(2, 0, 1).float().unsqueeze(0)           # 1,3k,Hp,Wp (CPU)
    state = base.repeat(n_samples, 1, 1, 1) if mode == "trajectory" else base

    preds, stds = [], []
    for step in range(n_future):
        outs = []
        for i in range(0, n_samples, batch):
            b = min(batch, n_samples - i)
            x = state[i:i + b].clone() if mode == "trajectory" else state.repeat(b, 1, 1, 1)
            x, sh, x_pre = _perturb(x, noise_std, max_shift, rng)
            y = G(x.to(device)).cpu()
            if getattr(G, "residual", False):
                y = y - (x - x_pre)[:, -3:]       # residual 模式: 扣掉經由「上一年」直通輸出的輸入噪聲
            outs.append(_unshift(y.clamp(-1, 1), sh))
        stack = torch.cat(outs, 0)                               # N,3,Hp,Wp
        if band_px > 0 or protect is not None:
            last = state[:, -3:]                                  # 乾淨的上一年 (1 或 N,3,Hp,Wp)
            if band_px > 0:
                from coastal_metrics import shore_band
                ref = last.median(0).values if last.shape[0] > 1 else last[0]
                w = torch.from_numpy(shore_band(ref.permute(1, 2, 0).numpy(), band_px, band_feather))
            else:
                w = torch.ones(stack.shape[2:])
            if protect is not None:
                hp, wp = stack.shape[2:]
                P = np.pad(protect, ((0, hp - protect.shape[0]), (0, wp - protect.shape[1])))
                w = w * (1 - torch.from_numpy(P).float())
            stack = w * stack + (1 - w) * last
            if bands is not None:
                bands.append(w.numpy()[:h0, :w0])
        a = stack.median(0).values if agg == "median" else stack.mean(0)
        s = stack.std(0).mean(0) if n_samples > 1 else torch.zeros(stack.shape[2:])

        if mode == "trajectory":
            state = torch.cat([state[:, 3:], stack], 1)
        else:
            state = torch.cat([state[:, 3:], a.unsqueeze(0)], 1)

        preds.append(a.permute(1, 2, 0).numpy()[:h0, :w0])
        stds.append(s.numpy()[:h0, :w0])
        print(f"  [rollout] step {step + 1}/{n_future}  mean std={float(s.mean()):.4f}")
    G.train()
    return preds, stds


# ----------------------------------------------------------------------------
# 5. 留出驗證: 模型 vs 持續性基準
# ----------------------------------------------------------------------------
def mae255(a, b):
    return float(np.abs(a - b).mean() * 127.5)   # 轉回 0-255 尺度


def validate(preds, ds, args):
    t0 = ds.train_end
    persist = ds.imgs[t0]
    rows = []
    print("\n===== 留出驗證 (MAE, 0-255 尺度, 越低越好) =====")
    print(f"{'year':>6} {'model':>9} {'persistence':>12}  winner")
    for i, p in enumerate(preds, 1):
        y = t0 + i
        if y not in ds.imgs:
            break
        m, b = mae255(p, ds.imgs[y]), mae255(persist, ds.imgs[y])
        rows.append({"year": y, "model_mae": m, "persistence_mae": b})
        print(f"{y:>6} {m:>9.3f} {b:>12.3f}  {'model' if m < b else 'persistence'}")
    if rows:
        mm = np.mean([r["model_mae"] for r in rows])
        bm = np.mean([r["persistence_mae"] for r in rows])
        print(f"{'mean':>6} {mm:>9.3f} {bm:>12.3f}")
        if mm >= bm:
            print("-> 模型沒有贏過「直接複製最後一年」, 目前沒有預測價值。")
        else:
            print("-> 模型優於持續性基準 (仍需再看岸線位置是否正確, MAE 只是整體像素誤差)。")
    with open(os.path.join(args.out_dir, "validation.json"), "w") as f:
        json.dump(rows, f, indent=2)


def save_img(arr, path):
    Image.fromarray(((arr + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(path)


# ----------------------------------------------------------------------------
# 6. 主程式
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--patch_w", type=int, default=384)
    p.add_argument("--patch_h", type=int, default=187)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--samples_per_epoch", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_resblocks", type=int, default=6)
    p.add_argument("--l1_weight", type=float, default=100.0)
    p.add_argument("--adv_weight", type=float, default=1.0, help="0 = 純 L1, 不用 GAN")
    p.add_argument("--residual", action="store_true",
                   help="殘差式生成: 輸出=上一年+變化量, 可大幅減少模糊與設施消失 (建議開啟)")
    p.add_argument("--flip", action="store_true", help="左右翻轉增強 (預設關)")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--holdout", type=int, default=0,
                   help=">0: 留出最後N年做驗證; 0: 全資料訓練並預測未來")
    p.add_argument("--n_future", type=int, default=12)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--mode", choices=["step", "trajectory"], default="trajectory")
    p.add_argument("--agg", choices=["median", "mean"], default="median")
    p.add_argument("--noise_std", type=float, default=0.03)
    p.add_argument("--max_shift", type=int, default=2)
    p.add_argument("--infer_batch", type=int, default=10)
    p.add_argument("--band_px", type=int, default=20,
                   help="只允許岸線附近多少像素內變動 (0=關閉, 整張圖都由模型重畫)")
    p.add_argument("--band_feather", type=int, default=4)
    p.add_argument("--protect_box", action="append", default=[],
                   help="強制保留區 x0,y0,x1,y1 (像素, 可重複指定), 該區域一律沿用上一年影像")
    p.add_argument("--protect_mask", default=None, help="強制保留區遮罩圖 (白色=保留)")
    p.add_argument("--mask_dir", default=None, help="歷年陸地遮罩資料夾 (校準岸線帶的分割)")
    p.add_argument("--mask_polarity", choices=["auto", "land_white", "land_black"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(f"[info] args={args}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    G, ds = train(args, device)

    if args.mask_dir:
        from coastal_metrics import setup_masks
        inv = {"auto": None, "land_white": False, "land_black": True}[args.mask_polarity]
        setup_masks(ds, args.mask_dir, ds.train_end, inv)
    init = [ds.imgs[y] for y in range(ds.train_end - ds.k + 1, ds.train_end + 1)]
    n_future = args.holdout if args.holdout > 0 else args.n_future
    protect = None
    if args.protect_box or args.protect_mask:
        from coastal_metrics import make_protect
        boxes = [tuple(int(float(t)) for t in b.split(",")) for b in args.protect_box]
        protect = make_protect((ds.h, ds.w), boxes, args.protect_mask)
    preds, stds = rollout(G, init, n_future, args.n_samples, args.mode,
                          args.noise_std, args.max_shift, args.infer_batch,
                          args.agg, device, args.seed,
                          band_px=args.band_px, band_feather=args.band_feather, protect=protect)

    for i, (pr, sd) in enumerate(zip(preds, stds), 1):
        y = ds.train_end + i
        save_img(pr, os.path.join(args.out_dir, f"pred_{y}.png"))
        s = (sd / (sd.max() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(s).save(os.path.join(args.out_dir, f"std_{y}.png"))

    if args.holdout > 0:
        validate(preds, ds, args)


if __name__ == "__main__":
    main()