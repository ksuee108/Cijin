"""
生成影像評估: KID / FID / LPIPS / MS-SSIM
==========================================
依賴:
    pip install torchmetrics torch-fidelity lpips
(首次執行會自動下載 Inception 與 AlexNet 權重, 需要網路)

指標與方向:
    FID      ↓  Fréchet Inception Distance    分佈層級, 比較「預測圖塊」與「真實圖塊」的特徵分佈
    KID      ↓  Kernel Inception Distance     同上, 無偏估計, 小樣本比 FID 穩定 (回報 mean ± std)
    LPIPS    ↓  Learned Perceptual Patch Sim. 逐張, 預測圖 vs 真實圖的感知距離
    MS-SSIM  ↑  Multi-Scale SSIM              逐張, 預測圖 vs 真實圖的多尺度結構相似度

FID/KID 的樣本來源:
    影像只有 1627x187, 一張圖只算 1 個樣本, FID 的協方差無法估計。
    因此把每張圖切成 187x187 的圖塊 (stride 預設 48), 以「圖塊集合」作為分佈。
    圖塊彼此重疊、並非獨立樣本, 數字只適合在同一資料、同一設定下做相對比較。

每個指標都同時對「模型預測」與「持續性基準」(直接複製訓練期最後一年) 計算。
"""
import csv
import json
import os

import numpy as np

MS_SSIM_BETAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


# ----------------------------------------------------------------------------
# 純 numpy 輔助
# ----------------------------------------------------------------------------
def extract_patches(arr, ps=None, stride=48):
    """arr: HxWx3 in [-1,1] -> N x ps x ps x 3 (沿寬度滑動)"""
    h, w, _ = arr.shape
    ps = ps or min(h, w)
    xs = list(range(0, w - ps + 1, stride))
    if xs[-1] != w - ps:
        xs.append(w - ps)
    return np.stack([arr[:ps, x:x + ps] for x in xs])


def ms_ssim_betas(h, w, kernel=11):
    """依影像尺寸決定可用尺度數 (需 min(h,w) > (kernel-1)*2^(n-1)), 權重重新正規化"""
    n = len(MS_SSIM_BETAS)
    m = min(h, w)
    while n > 1 and m <= (kernel - 1) * 2 ** (n - 1):
        n -= 1
    b = np.array(MS_SSIM_BETAS[:n])
    return tuple((b / b.sum()).tolist())


def _to_tensor(arr, device):
    """HxWx3 或 NxHxWx3 in [-1,1] -> N,3,H,W float tensor"""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.permute(0, 3, 1, 2).to(device)


# ----------------------------------------------------------------------------
# 評估器
# ----------------------------------------------------------------------------
class GenMetrics:
    def __init__(self, device="cpu", fid_feature=2048, patch_stride=48,
                 kid_subsets=100, kid_subset_size=50):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.fid_feature = fid_feature
        self.stride = patch_stride
        self.kid_subsets = kid_subsets
        self.kid_subset_size = kid_subset_size
        self._lpips = None

    # ---- 逐張指標 ----
    def lpips(self, pred, true):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        with self.torch.no_grad():
            d = self._lpips(_to_tensor(pred, self.device), _to_tensor(true, self.device))
        return float(d.mean())

    def ms_ssim(self, pred, true):
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
        h, w, _ = pred.shape
        metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=11, betas=ms_ssim_betas(h, w)).to(self.device)
        p = (_to_tensor(pred, self.device) + 1) / 2
        t = (_to_tensor(true, self.device) + 1) / 2
        with self.torch.no_grad():
            return float(metric(p.clamp(0, 1), t.clamp(0, 1)))

    # ---- 分佈指標 ----
    def _feed(self, metric, patches, real):
        x = (_to_tensor(patches, self.device) + 1) / 2          # [0,1]
        for i in range(0, len(x), 32):
            metric.update(x[i:i + 32].clamp(0, 1), real=real)

    def fid(self, fake_patches, real_patches):
        from torchmetrics.image.fid import FrechetInceptionDistance
        m = FrechetInceptionDistance(feature=self.fid_feature, normalize=True).to(self.device)
        self._feed(m, real_patches, True)
        self._feed(m, fake_patches, False)
        return float(m.compute())

    def kid(self, fake_patches, real_patches):
        from torchmetrics.image.kid import KernelInceptionDistance
        n = min(len(fake_patches), len(real_patches))
        # subset 不可等於 n, 否則每個子集都相同, KID 的 std 會是 0 而失去意義
        subset = max(2, min(self.kid_subset_size, n // 2))
        m = KernelInceptionDistance(feature=self.fid_feature, subsets=self.kid_subsets,
                                    subset_size=subset, normalize=True).to(self.device)
        self._feed(m, real_patches, True)
        self._feed(m, fake_patches, False)
        mean, std = m.compute()
        return float(mean), float(std)


# ----------------------------------------------------------------------------
# 取代 validate(): 在 main() 裡呼叫 validate_gen_metrics(preds, ds, args, device)
# ----------------------------------------------------------------------------
def validate_gen_metrics(preds, ds, args, device="cpu", fid_feature=2048):
    gm = GenMetrics(device=device, fid_feature=fid_feature)
    t0 = ds.train_end
    ref = ds.imgs[t0]
    years = [t0 + i for i in range(1, len(preds) + 1) if (t0 + i) in ds.imgs]
    if not years:
        print("[gen_metrics] 沒有可比對的真實年份")
        return

    def dist_row(fake, real):
        km, ks = gm.kid(fake, real)
        return {"FID": gm.fid(fake, real), "KID_mean": km, "KID_std": ks}

    rows = []                                    # 每年、每個對象一列 (含 FID/KID)
    fake_all, base_all, real_all = [], [], []
    for y, p in zip(years, preds):
        true = ds.imgs[y]
        fp = extract_patches(p, stride=gm.stride)
        bp = extract_patches(ref, stride=gm.stride)
        rp = extract_patches(true, stride=gm.stride)
        fake_all.append(fp); base_all.append(bp); real_all.append(rp)
        for who, img, patches in (("model", p, fp), ("persistence", ref, bp)):
            rows.append({"year": y, "who": who,
                         "LPIPS": gm.lpips(img, true),
                         "MS-SSIM": gm.ms_ssim(img, true),
                         **dist_row(patches, rp),
                         "n_patches": int(len(patches))})
    fake_all, base_all, real_all = (np.concatenate(v) for v in (fake_all, base_all, real_all))

    # 全部年份合併 (圖塊數較多, FID/KID 較穩定)
    overall = {}
    for who, f in (("model", fake_all), ("persistence", base_all)):
        overall[who] = {"LPIPS": float(np.mean([r["LPIPS"] for r in rows if r["who"] == who])),
                        "MS-SSIM": float(np.mean([r["MS-SSIM"] for r in rows if r["who"] == who])),
                        **dist_row(f, real_all), "n_patches": int(len(f))}
        rows.append({"year": "all", "who": who, **overall[who]})

    # 存檔
    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["year", "who", "FID", "KID_mean", "KID_std", "LPIPS", "MS-SSIM", "n_patches"]
    with open(os.path.join(args.out_dir, "gen_metrics_per_year.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "gen_metrics.json"), "w") as f:
        json.dump({"rows": rows, "overall": overall, "fid_feature": fid_feature}, f, indent=2)

    # 列印 (FID / KID 逐年與合計)
    def get(y, who, k):
        return next(r[k] for r in rows if r["year"] == y and r["who"] == who)

    print(f"\n===== FID ↓ / KID ↓  (Inception-{fid_feature}, 圖塊 stride={gm.stride}) =====")
    print(f"{'year':>6} | {'FID model':>10} {'FID persist':>12} | {'KID model':>20} {'KID persist':>20} | patches")
    for y in years + ["all"]:
        km, ks = get(y, "model", "KID_mean"), get(y, "model", "KID_std")
        bm, bs = get(y, "persistence", "KID_mean"), get(y, "persistence", "KID_std")
        print(f"{str(y):>6} | {get(y,'model','FID'):>10.3f} {get(y,'persistence','FID'):>12.3f} | "
              f"{km:>13.5f}±{ks:.5f} {bm:>13.5f}±{bs:.5f} | {get(y,'model','n_patches')}")

    print("\n===== LPIPS ↓ / MS-SSIM ↑ =====")
    print(f"{'year':>6} | {'LPIPS model':>12} {'LPIPS persist':>14} | {'MSSSIM model':>13} {'MSSSIM persist':>15}")
    for y in years + ["all"]:
        print(f"{str(y):>6} | {get(y,'model','LPIPS'):>12.4f} {get(y,'persistence','LPIPS'):>14.4f} | "
              f"{get(y,'model','MS-SSIM'):>13.4f} {get(y,'persistence','MS-SSIM'):>15.4f}")

    print("\n解讀:")
    print(" - 逐年的 FID/KID 只有約 30 個圖塊, 很不穩定, 請以 'all' (全部年份合併) 為主, 逐年僅供參考")
    print(" - FID/KID 衡量「像不像真實衛星影像」, 不管岸線位置對不對")
    print(" - 使用帶狀保留時, 帶外像素直接沿用上一年影像, 整張圖計算的指標會被拉近持續性基準, 並非模型更準")
    print(f"\n輸出: {args.out_dir}/gen_metrics_per_year.csv (含 FID/KID/LPIPS/MS-SSIM), gen_metrics.json")