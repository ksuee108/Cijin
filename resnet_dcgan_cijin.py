"""
ResNet-DCGAN 海岸線時序預測 — 旗津 (cijin) 資料集版
基於 resnet_dcgan_coastal_v2.py 修改。

資料結構 (cijin.zip 解壓後):
    cijin/
      海水/2013.jpg ... 2026.jpg      (缺 2020)
      風車/2013.jpg ... 2026.jpg
  每個子資料夾 = 一個地點 (site), 檔名 = 年份, 影像 1024x768。

相對 v2 的主要修改:
  1. 資料集支援多地點子資料夾; 也相容原本「年份圖片直接放在 data_dir」的單地點結構。
  2. 只使用「真正連續年份」的序列。海水缺 2020, 原版會把 2019->2021 當成連續一年,
     這裡會自動排除跨越缺年的視窗 (k=3、不留出時: 海水 7 個視窗、風車 11 個;
     --holdout 3 時: 海水 4 個、風車 8 個)。
  3. 留出驗證 (--holdout N) 改為「每個地點各自留出最後 N 張」, 預測年數用
     (最後一年 - train_end) 計算, 因此遇到缺年也不會錯位。
  4. 新增色彩校正 (--color_match meanstd, 預設開): 各年份衛星影像色調差很大
     (2019 偏暗、2026 偏藍), 不處理的話 L1 會被迫去學無法預測的色偏。
     參考色彩統計只用訓練年份計算, 不會洩漏留出年份的資訊。
  5. 預設值改成適合 1024x768 / 影像很少的情況:
     patch 256x256, infer_batch 2, n_samples 20, band_px 0 (見下方說明)。
  6. --residual 預設開啟 (用 --no-residual 關閉)。
  7. 每個地點各自輸出到 out_dir/<地點>/, 留出驗證時多存一張
     [最後訓練年 | 預測 | 真實] 對照圖。
  8. 移除 --mask_dir (setup_masks 依賴原本的 dataset 內部結構, 多地點版無法直接套用)。
     --protect_box / --protect_mask / --band_px 仍透過 coastal_metrics 使用,
     需要 coastal_metrics.py 放在同一資料夾; band_px 預設改 0 所以不必要時不會 import。

用法:
  # (A) 留出驗證: 每個地點留最後 3 年 (海水 2024-2026 / 風車 2024-2026)
  python resnet_dcgan_cijin.py --data_dir ./cijin --out_dir ./val --holdout 3 --epochs 150

  # (B) 全資料訓練並預測未來 12 年
  python resnet_dcgan_cijin.py --data_dir ./cijin --out_dir ./future --n_future 12 --epochs 150

  # 只跑單一地點
  python resnet_dcgan_cijin.py --data_dir ./cijin --sites 風車 --holdout 3
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

IMG_EXTS = ("jpg", "jpeg", "png", "tif", "tiff")


# ----------------------------------------------------------------------------
# 1. 資料集
# ----------------------------------------------------------------------------
def _find_year_paths(folder):
    """回傳 {年份: 路徑}; 檔名不是純數字的檔案會被略過。"""
    paths = []
    for ext in IMG_EXTS:
        paths += glob(os.path.join(folder, f"*.{ext}"))
        paths += glob(os.path.join(folder, f"*.{ext.upper()}"))
    year_path = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            year_path[int(name)] = p
        except ValueError:
            continue
    return year_path


def _discover_sites(data_dir, wanted=None):
    """子資料夾 (內含年份圖片) = 地點; 若 data_dir 本身就有年份圖片, 視為單一地點。"""
    sites = {}
    if _find_year_paths(data_dir):
        sites[os.path.basename(os.path.abspath(data_dir)) or "site"] = data_dir
    for name in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, name)
        if os.path.isdir(sub) and _find_year_paths(sub):
            sites[name] = sub
    if wanted:
        missing = [s for s in wanted if s not in sites]
        assert not missing, f"找不到地點 {missing}, 現有: {list(sites)}"
        sites = {s: sites[s] for s in wanted}
    assert sites, f"{data_dir} 底下找不到「年份.jpg」格式的影像"
    return sites


class CoastalSequenceDataset(Dataset):
    def __init__(self, data_dir, sites=None, k=3, patch_w=256, patch_h=256,
                 samples_per_epoch=800, flip=False, holdout=0, color_match="meanstd"):
        self.k = k
        self.samples_per_epoch = samples_per_epoch
        self.flip = flip
        self.patch_w, self.patch_h = patch_w, patch_h

        self.sites = {}       # site -> dict(imgs, years, train_end, h, w)
        self.windows = []     # [(site, start_index)]
        for name, folder in _discover_sites(data_dir, sites).items():
            year_path = _find_year_paths(folder)
            years = sorted(year_path)
            assert len(years) >= k + 1 + holdout, \
                f"[{name}] 影像張數不足 ({len(years)} 張, 需要 >= {k + 1 + holdout})"
            missing = [y for y in range(years[0], years[-1] + 1) if y not in year_path]
            if missing:
                print(f"[warn] {name}: 缺少年份 {missing}, 跨越缺年的序列不會被使用")

            imgs = {}
            for y in years:
                arr = np.asarray(Image.open(year_path[y]).convert("RGB")).astype(np.float32)
                imgs[y] = arr / 127.5 - 1.0
            shapes = {v.shape for v in imgs.values()}
            assert len(shapes) == 1, f"[{name}] 影像尺寸不一致: {shapes}"
            h, w, _ = imgs[years[0]].shape

            train_end = years[-1 - holdout]
            starts = [i for i in range(len(years) - k)
                      if years[i + k] - years[i] == k      # 連續 k+1 年
                      and years[i + k] <= train_end]       # 目標年不得超過 train_end
            assert starts, f"[{name}] 沒有可用的連續訓練序列, 請減小 k 或 holdout"

            # rollout 的初始畫面需要 train_end 之前連續 k 年
            init_years = list(range(train_end - k + 1, train_end + 1))
            assert all(y in imgs for y in init_years), \
                f"[{name}] train_end={train_end} 之前沒有連續 {k} 年, 無法做預測起點"

            if color_match == "meanstd":
                train_years = [y for y in years if y <= train_end]
                stats = [(imgs[y].mean((0, 1)), imgs[y].std((0, 1)) + 1e-6) for y in train_years]
                ref_m = np.mean([s[0] for s in stats], 0)
                ref_s = np.mean([s[1] for s in stats], 0)
                for y in years:
                    m, s = imgs[y].mean((0, 1)), imgs[y].std((0, 1)) + 1e-6
                    imgs[y] = ((imgs[y] - m) / s * ref_s + ref_m).clip(-1, 1).astype(np.float32)

            self.sites[name] = dict(imgs=imgs, years=years, train_end=train_end, h=h, w=w,
                                    starts=starts, init_years=init_years)
            self.windows += [(name, i) for i in starts]
            print(f"[data] {name}: years={years[0]}-{years[-1]} ({len(years)} 張), "
                  f"train_end={train_end}, 訓練視窗={len(starts)}, size={w}x{h}")
        print(f"[data] 共 {len(self.sites)} 個地點, 訓練視窗合計 {len(self.windows)}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        name, i = random.choice(self.windows)
        s = self.sites[name]
        seq = s["years"][i: i + self.k + 1]
        pw, ph = min(self.patch_w, s["w"]), min(self.patch_h, s["h"])
        x = random.randint(0, s["w"] - pw)
        y = random.randint(0, s["h"] - ph)
        do_flip = self.flip and random.random() < 0.5

        def patch(year):
            a = s["imgs"][year][y:y + ph, x:x + pw, :]
            return a[:, ::-1, :].copy() if do_flip else a

        past = np.concatenate([patch(t) for t in seq[:-1]], axis=-1)
        tgt = patch(seq[-1])
        return (torch.from_numpy(past).permute(2, 0, 1).float(),
                torch.from_numpy(tgt).permute(2, 0, 1).float())


# ----------------------------------------------------------------------------
# 2. 模型 (與 v2 相同)
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
# 3. 訓練 (與 v2 相同, 另外存 config.json 方便日後載入)
# ----------------------------------------------------------------------------
def train(args, device):
    ds = CoastalSequenceDataset(args.data_dir, sites=args.sites, k=args.k,
                                patch_w=args.patch_w, patch_h=args.patch_h,
                                samples_per_epoch=args.samples_per_epoch,
                                flip=args.flip, holdout=args.holdout,
                                color_match=args.color_match)
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

        if epoch % args.log_every == 0 or epoch == args.epochs or epoch == 1:
            print(f"[epoch {epoch:4d}/{args.epochs}] D={loss_D.item():.4f} "
                  f"G_adv={loss_adv.item():.4f} G_l1={loss_l1.item():.4f}")
    torch.save(G.state_dict(), os.path.join(args.out_dir, "generator.pt"))
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(k=args.k, ngf=args.ngf, n_resblocks=args.n_resblocks,
                       residual=args.residual, color_match=args.color_match,
                       sites=list(ds.sites)), f, indent=2, ensure_ascii=False)
    return G, ds


# ----------------------------------------------------------------------------
# 4. 集成式自回歸預測 (與 v2 相同)
# ----------------------------------------------------------------------------
def _pad4(arr):
    h, w, _ = arr.shape
    return np.pad(arr, ((0, (4 - h % 4) % 4), (0, (4 - w % 4) % 4), (0, 0)), mode="reflect")


def _perturb(x, noise_std, max_shift, rng):
    """G 是確定性的, 以輸入雜訊 + 小幅平移製造樣本間差異。
    回傳 (擾動後的輸入, 每個樣本的平移量, 加雜訊前的輸入); 輸出必須用 _unshift 平移回來,
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
# 5. 留出驗證: 模型 vs 持續性基準 (逐地點; 遇到缺年自動跳過)
# ----------------------------------------------------------------------------
def mae255(a, b):
    return float(np.abs(a - b).mean() * 127.5)   # 轉回 0-255 尺度


def validate(preds, site, out_dir):
    imgs, t0 = site["imgs"], site["train_end"]
    persist = imgs[t0]
    rows = []
    print(f"{'year':>6} {'model':>9} {'persistence':>12}  winner")
    for i, p in enumerate(preds, 1):
        y = t0 + i
        if y not in imgs:          # 缺年 (例如 2020), 沒有真實影像可比
            continue
        m, b = mae255(p, imgs[y]), mae255(persist, imgs[y])
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
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(rows, f, indent=2)


def save_img(arr, path):
    Image.fromarray(((arr + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(path)


def save_compare(last_train, pred, truth, path):
    """[最後訓練年 | 預測 | 真實] 橫向對照圖。"""
    save_img(np.concatenate([last_train, pred, truth], axis=1), path)


# ----------------------------------------------------------------------------
# 6. 主程式
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="cijin 資料夾 (底下是 海水/ 風車/ 子資料夾)")
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--sites", nargs="*", default=None, help="只使用指定地點, 例如 --sites 風車 海水")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--patch_w", type=int, default=256)
    p.add_argument("--patch_h", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--samples_per_epoch", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_resblocks", type=int, default=6)
    p.add_argument("--l1_weight", type=float, default=100.0)
    p.add_argument("--adv_weight", type=float, default=1.0, help="0 = 純 L1, 不用 GAN")
    p.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                   help="殘差式生成: 輸出=上一年+變化量 (預設開, 用 --no-residual 關閉)")
    p.add_argument("--color_match", choices=["meanstd", "none"], default="meanstd",
                   help="把每年影像的各通道均值/標準差對齊到訓練年份的平均, 消除年際色偏")
    p.add_argument("--flip", action="store_true", help="左右翻轉增強 (預設關)")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--holdout", type=int, default=0,
                   help=">0: 每個地點留出最後N張做驗證; 0: 全資料訓練並預測未來")
    p.add_argument("--n_future", type=int, default=12)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--mode", choices=["step", "trajectory"], default="trajectory")
    p.add_argument("--agg", choices=["median", "mean"], default="median")
    p.add_argument("--noise_std", type=float, default=0.03)
    p.add_argument("--max_shift", type=int, default=2)
    p.add_argument("--infer_batch", type=int, default=2, help="1024x768 整張推論很吃記憶體, 不夠就調小")
    p.add_argument("--band_px", type=int, default=0,
                   help="只允許岸線附近多少像素內變動 (0=關閉; >0 需要 coastal_metrics.py)")
    p.add_argument("--band_feather", type=int, default=4)
    p.add_argument("--protect_box", action="append", default=[],
                   help="強制保留區 x0,y0,x1,y1 (像素, 可重複指定), 該區域一律沿用上一年影像")
    p.add_argument("--protect_mask", default=None, help="強制保留區遮罩圖 (白色=保留)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(f"[info] args={args}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    G, ds = train(args, device)

    protect = None
    if args.protect_box or args.protect_mask:
        from coastal_metrics import make_protect

    for name, site in ds.sites.items():
        print(f"\n===== 預測地點: {name} =====")
        site_dir = os.path.join(args.out_dir, name)
        os.makedirs(site_dir, exist_ok=True)
        t0 = site["train_end"]
        init = [site["imgs"][y] for y in site["init_years"]]
        n_future = (site["years"][-1] - t0) if args.holdout > 0 else args.n_future

        if args.protect_box or args.protect_mask:
            boxes = [tuple(int(float(t)) for t in b.split(",")) for b in args.protect_box]
            protect = make_protect((site["h"], site["w"]), boxes, args.protect_mask)

        preds, stds = rollout(G, init, n_future, args.n_samples, args.mode,
                              args.noise_std, args.max_shift, args.infer_batch,
                              args.agg, device, args.seed,
                              band_px=args.band_px, band_feather=args.band_feather,
                              protect=protect)

        for i, (pr, sd) in enumerate(zip(preds, stds), 1):
            y = t0 + i
            save_img(pr, os.path.join(site_dir, f"pred_{y}.png"))
            s = (sd / (sd.max() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(s).save(os.path.join(site_dir, f"std_{y}.png"))
            if args.holdout > 0 and y in site["imgs"]:
                save_compare(site["imgs"][t0], pr, site["imgs"][y],
                             os.path.join(site_dir, f"compare_{y}.png"))

        if args.holdout > 0:
            print(f"\n----- 留出驗證 [{name}] (MAE, 0-255 尺度, 越低越好) -----")
            validate(preds, site, site_dir)


if __name__ == "__main__":
    main()
