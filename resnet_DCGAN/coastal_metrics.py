"""
海岸線預測評估指標
==================
依賴: pip install scikit-image scipy

指標分三類 (全部同時對「模型」與「持續性基準」計算, 方便比較):
  A. 影像品質:   MAE / RMSE / PSNR / SSIM
  B. 陸水分類:   水體 IoU / F1 / 準確率  (水體由顏色啟發式分割, 見 water_mask)
  C. 岸線與變遷: 岸線 MAE / RMSE (px 或 m)、最大誤差、
                 變遷區域 F1 (是否抓到真的有變化的地方)、
                 位移相關係數 (預測的位移方向是否與真實一致)、
                 岸線技能分數 SS = 1 - MSE_model / MSE_persistence (>0 才贏基準)

重要: 持續性基準 (直接複製最後一年) 在 A 類指標上通常很強, 因為相鄰年份幾乎一樣。
      真正能判斷模型有沒有學到演變的是 C 類 (尤其 變遷F1 與 技能分數)。

water_mask 是顏色啟發式, 只是近似; 若你有正式的陸/水分割遮罩, 請替換該函式。
"""
import csv
import json
import os
from glob import glob

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity


# ----------------------------------------------------------------------------
# 基礎
# ----------------------------------------------------------------------------
def to_u8(arr):
    """[-1,1] float -> uint8"""
    return ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


SEG = {"br": 28.0, "lum": 120.0}      # 顏色分割參數; 有真實遮罩時會由 setup_masks 校準


def clean_sea(m, min_area=150, smooth=False):
    """保留與影像底邊相連的海, 並填補海中小洞 (礁石、浪花)。smooth=True 時先做形態學平滑。"""
    m = m.astype(bool)
    if smooth:
        # 先以邊緣複製補邊, 避免形態學 closing 在影像邊界把底邊侵蝕掉
        m = np.pad(m, 6, mode="edge")
        m = ndi.binary_opening(m, structure=np.ones((3, 3)))
        m = ndi.binary_closing(m, structure=np.ones((5, 5)))
        m = m[6:-6, 6:-6]
    lab, n = ndi.label(m)
    keep = np.zeros_like(m)
    for i in np.unique(lab[-1, :]):
        if i != 0:
            keep |= lab == i
    m = keep
    holes, n = ndi.label(~m)
    if n:
        sizes = ndi.sum(~m, holes, range(1, n + 1))
        for i, sz in enumerate(sizes, 1):
            if sz < min_area:
                m[holes == i] = True
    return m


def water_mask(u8, min_area=150, br=None, lum=None):
    """
    海水顏色分割 (用於「沒有真實遮罩」的影像, 例如模型預測圖):
      藍 - 紅 > br 且亮度 < lum, 再只保留與影像底邊相連的水體。
    br/lum 預設取 SEG; 有歷年遮罩時由 setup_masks 校準成最貼近你遮罩的值。
    前提: 影像下方是海、上方是陸。
    """
    br = SEG["br"] if br is None else br
    lm = SEG["lum"] if lum is None else lum
    r, g, b = [u8[..., i].astype(np.float32) for i in range(3)]
    lu = (r + g + b) / 3.0
    return clean_sea((b - r > br) & (lu < lm), min_area, smooth=True)


def load_sea_masks(mask_dir, years, shape, invert=None, verbose=True):
    """
    讀取歷年陸地遮罩 -> {year: 海水 bool 遮罩 (HxW)}。
    檔名: {year}.png/jpg/tif... (或檔名含年份)。二值化後以影像版面自動判斷極性
    (陸在上方、海在下方), 也可用 invert 強制: False=遮罩亮色為陸, True=遮罩亮色為海。
    """
    out, shown = {}, False
    for y in years:
        cands = []
        for ext in ("png", "jpg", "jpeg", "tif", "tiff", "bmp"):
            cands += glob(os.path.join(mask_dir, f"{y}.{ext}"))
        if not cands:
            for ext in ("png", "jpg", "jpeg", "tif", "tiff", "bmp"):
                cands += sorted(glob(os.path.join(mask_dir, f"*{y}*.{ext}")))
        if not cands:
            continue
        im = Image.open(cands[0]).convert("L")
        if (im.size[1], im.size[0]) != tuple(shape):
            if verbose:
                print(f"[mask] {y}: 遮罩尺寸 {im.size[::-1]} 與影像 {tuple(shape)} 不同, 以最近鄰縮放")
            im = im.resize((shape[1], shape[0]), Image.NEAREST)
        arr = np.asarray(im)
        bright = arr > (127 if arr.max() > 1 else 0)
        top, bot = bright[:8].mean(), bright[-8:].mean()
        inv = (top < bot) if invert is None else bool(invert)
        land = ~bright if inv else bright
        out[y] = clean_sea(~land, smooth=False)
        if verbose and not shown:
            shown = True
            print(f"[mask] 範例 {os.path.basename(cands[0])}: 唯一值={np.unique(arr)[:6].tolist()}, "
                  f"頂端亮色比例={top:.2f}, 底端={bot:.2f} -> 亮色={'海' if inv else '陸'} "
                  f"({'自動判斷' if invert is None else '手動指定'}), 海占比={out[y].mean():.2f}")
    if verbose:
        miss = [y for y in years if y not in out]
        print(f"[mask] 載入 {len(out)}/{len(years)} 年遮罩" + (f", 缺少: {miss}" if miss else ""))
    return out


def fit_seg_params(imgs, sea_masks, years, n_use=4):
    """在有真實遮罩的年份上, 格點搜尋最貼近遮罩的顏色分割參數 (br, lum), 回傳 (br, lum, mean_IoU)"""
    ys = [y for y in years if y in sea_masks and y in imgs]
    if not ys:
        return SEG["br"], SEG["lum"], float("nan")
    idx = np.unique(np.linspace(0, len(ys) - 1, min(n_use, len(ys))).round().astype(int))
    ys = [ys[i] for i in idx]
    best = (SEG["br"], SEG["lum"], -1.0)
    for br in (12, 20, 28, 36, 44):
        for lum in (90, 105, 120, 135, 150):
            ious = []
            for y in ys:
                pm = water_mask(to_u8(imgs[y]), br=br, lum=lum)
                t = sea_masks[y]
                u = np.logical_or(pm, t).sum()
                ious.append(np.logical_and(pm, t).sum() / u if u else 1.0)
            if np.mean(ious) > best[2]:
                best = (br, lum, float(np.mean(ious)))
    return best


def setup_masks(ds, mask_dir, upto_year, invert=None):
    """
    載入歷年遮罩, 只用 <= upto_year 的年份校準分割參數 (避免用到驗證年份),
    結果存在 ds.sea_masks, 並更新 water_mask 預設參數 (岸線帶、預測圖的分割都會用到)。
    """
    ds.sea_masks = load_sea_masks(mask_dir, ds.years, (ds.h, ds.w), invert)
    fit_years = [y for y in ds.years if y <= upto_year]
    br, lum, iou = fit_seg_params(ds.imgs, ds.sea_masks, fit_years)
    SEG["br"], SEG["lum"] = float(br), float(lum)
    print(f"[mask] 顏色分割校準: br={br}, lum={lum}, 對真實遮罩的平均水體 IoU={iou:.3f}")
    if iou == iou and iou < 0.90:
        print("[warn] 顏色分割與你的遮罩差異偏大 (IoU<0.90): 預測圖的岸線量測會有明顯誤差,"
              " 建議改成讓模型直接預測遮罩, 而不是事後分割預測圖。")


def shoreline(mask):
    """
    逐欄取岸線: 由上往下第一個水體像素的列號 (影像上方為陸、下方為海)。
    該欄無水體則記為 H。回傳 shape (W,) 的 float 陣列。
    """
    h, w = mask.shape
    first = np.where(mask.any(0), mask.argmax(0), h)
    return first.astype(np.float32)


# ----------------------------------------------------------------------------
# 指標
# ----------------------------------------------------------------------------
def image_metrics(a_u8, b_u8):
    d = a_u8.astype(np.float32) - b_u8.astype(np.float32)
    mse = float((d ** 2).mean())
    return {
        "MAE": float(np.abs(d).mean()),
        "RMSE": float(np.sqrt(mse)),
        "PSNR": float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf"),
        "SSIM": float(structural_similarity(a_u8, b_u8, channel_axis=2, data_range=255)),
    }


def mask_metrics(pm, tm):
    inter = np.logical_and(pm, tm).sum()
    union = np.logical_or(pm, tm).sum()
    tp, fp, fn = inter, np.logical_and(pm, ~tm).sum(), np.logical_and(~pm, tm).sum()
    return {
        "water_IoU": float(inter / union) if union else float("nan"),
        "water_F1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "land_water_acc": float((pm == tm).mean()),
    }


def f1(pred, true):
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d else float("nan")


def shoreline_metrics(sp, st, s0, px_m=None):
    """sp=預測岸線, st=真實岸線, s0=參考(訓練期最後一年)岸線, 單位 px"""
    e = sp - st
    k = px_m if px_m else 1.0
    unit = "m" if px_m else "px"
    out = {
        f"shore_MAE_{unit}": float(np.abs(e).mean() * k),
        f"shore_RMSE_{unit}": float(np.sqrt((e ** 2).mean()) * k),
        f"shore_max_{unit}": float(np.abs(e).max() * k),
    }
    d_true, d_pred = st - s0, sp - s0                      # 相對參考年的位移
    out["disp_true_mean_" + unit] = float(d_true.mean() * k)
    out["disp_pred_mean_" + unit] = float(d_pred.mean() * k)
    if d_true.std() > 1e-6 and d_pred.std() > 1e-6:
        out["disp_corr"] = float(np.corrcoef(d_pred, d_true)[0, 1])
    else:
        out["disp_corr"] = float("nan")
    mse_m = float((e ** 2).mean())
    mse_p = float(((s0 - st) ** 2).mean())                 # 持續性基準 = 岸線不變
    out["skill_vs_persist"] = float(1 - mse_m / mse_p) if mse_p > 0 else float("nan")
    return out


def evaluate_one(pred_f, true_f, ref_f, px_m=None, mask_true=None, mask_ref=None):
    """
    pred_f/true_f/ref_f: HxWx3 in [-1,1]
    mask_true / mask_ref: 真實年份與參考年的海水遮罩 (有的話取代顏色分割, 作為「真實岸線」)。
    預測圖沒有真實遮罩, 一律用顏色分割 (已用你的遮罩校準);
    seg_floor = 對「真實影像」做同樣的分割 vs 真實遮罩的誤差, 即量測本身的誤差下限。
    回傳 {"model": {...}, "persistence": {...}, ("seg_floor": {...})}
    """
    P, T, R = to_u8(pred_f), to_u8(true_f), to_u8(ref_f)
    use_gt = mask_true is not None and mask_ref is not None
    mT = mask_true if use_gt else water_mask(T)
    mR = mask_ref if use_gt else water_mask(R)
    mP = water_mask(P)
    sP, sT, sR = shoreline(mP), shoreline(mT), shoreline(mR)

    res = {}
    for name, img, m, s in (("model", P, mP, sP), ("persistence", R, mR, sR)):
        r = {}
        r.update(image_metrics(img, T))
        r.update(mask_metrics(m, mT))
        r.update(shoreline_metrics(s, sT, sR, px_m))
        # 變遷區域: 相對參考年有改變的水陸像素
        true_change, pred_change = np.logical_xor(mT, mR), np.logical_xor(m, mR)
        r["change_F1"] = f1(pred_change, true_change) if true_change.any() else float("nan")
        res[name] = r
    if use_gt:
        mTs = water_mask(T)
        e = (shoreline(mTs) - sT) * (px_m if px_m else 1.0)
        unit = "m" if px_m else "px"
        fl = dict(mask_metrics(mTs, mT))
        fl.update({f"shore_MAE_{unit}": float(np.abs(e).mean()),
                   f"shore_RMSE_{unit}": float(np.sqrt((e ** 2).mean())),
                   f"shore_max_{unit}": float(np.abs(e).max())})
        res["seg_floor"] = fl
    res["_masks"] = (mR, mT, mP)
    res["_shore"] = (sR, sT, sP)
    return res


# ----------------------------------------------------------------------------
# 疊圖: 真實影像 + 三條岸線 (黃=參考年, 綠=真實, 紅=預測)
# ----------------------------------------------------------------------------
def save_overlay(true_f, shores, path):
    sR, sT, sP = shores
    img = Image.fromarray(to_u8(true_f))
    d = ImageDraw.Draw(img)
    for s, col in ((sR, (255, 255, 0)), (sT, (0, 255, 0)), (sP, (255, 0, 0))):
        pts = [(x, float(y)) for x, y in enumerate(s)]
        d.line(pts, fill=col, width=1)
    img.save(path)


# ----------------------------------------------------------------------------
# 取代 v2 的 validate(): 在 main() 裡改成 validate_with_metrics(preds, ds, args)
# ----------------------------------------------------------------------------
KEYS = ["MAE", "RMSE", "PSNR", "SSIM", "water_IoU", "water_F1", "land_water_acc"]


def validate_with_metrics(preds, ds, args, px_m=None):
    t0 = ds.train_end
    ref = ds.imgs[t0]
    unit = "m" if px_m else "px"
    masks = getattr(ds, "sea_masks", None) or {}
    rows, used_gt = [], False
    for i, p in enumerate(preds, 1):
        y = t0 + i
        if y not in ds.imgs:
            break
        mt, mr = masks.get(y), masks.get(t0)
        if masks and (mt is None or mr is None):
            print(f"[warn] {y} 或起點 {t0} 缺少遮罩, 這一年改用顏色分割")
            mt = mr = None
        used_gt = used_gt or (mt is not None)
        r = evaluate_one(p, ds.imgs[y], ref, px_m, mask_true=mt, mask_ref=mr)
        save_overlay(ds.imgs[y], r["_shore"], os.path.join(args.out_dir, f"overlay_{y}.png"))
        for who in ("model", "persistence"):
            rows.append({"year": y, "who": who, **r[who]})
        if "seg_floor" in r:
            fl = {k: float("nan") for k in r["model"]}
            fl.update(r["seg_floor"])
            rows.append({"year": y, "who": "seg_floor", **fl})

    if not rows:
        print("[metrics] 沒有可比對的真實年份")
        return

    cols = ["year", "who"] + [k for k in rows[0] if k not in ("year", "who")]
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    def mean_of(who, key):
        v = [r[key] for r in rows if r["who"] == who and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    show = KEYS + [f"shore_MAE_{unit}", f"shore_RMSE_{unit}", f"shore_max_{unit}", "disp_corr",
                   "change_F1", "skill_vs_persist"]
    fmt = lambda v: f"{v:>14.4f}" if v == v else f"{'-':>14}"
    print("\n===== 留出驗證指標 (各年平均) =====")
    print(f"真實岸線來源: {'你的歷年陸地遮罩' if used_gt else '顏色分割 (近似)'}")
    head = f"{'metric':<20}{'model':>14}{'persistence':>14}" + (f"{'seg_floor':>14}" if used_gt else "")
    print(head)
    for k in show:
        line = f"{k:<20}{fmt(mean_of('model', k))}{fmt(mean_of('persistence', k))}"
        if used_gt:
            line += fmt(mean_of("seg_floor", k))
        print(line)

    print("\n解讀:")
    print(" - MAE/RMSE/PSNR/SSIM: 影像整體相似度; 持續性基準通常很強, 贏不了不代表沒救")
    print(" - water_IoU / shore_*: 陸水與岸線位置; 誤差越低越好 (IoU/F1/acc 越高越好)")
    print(" - change_F1: 有沒有抓到真正發生變化的區域 (最能反映是否學到演變)")
    print(" - disp_corr: 預測位移與真實位移的相關, >0 方向一致; 接近 0 = 隨機")
    print(" - skill_vs_persist: >0 才贏過「岸線不變」, <=0 代表沒有預測價值")
    if used_gt:
        print(" - seg_floor: 對「真實影像」做同樣的顏色分割與真實遮罩的差距, 是預測圖岸線量測的誤差下限;")
        print("   模型的岸線誤差若與 seg_floor 同量級, 差異可能只是分割誤差, 不是預測誤差")
        print(" - persistence 的岸線直接取自真實遮罩 (無分割誤差), 模型則含分割誤差, 對模型略不利")
    print(f"\n輸出: {args.out_dir}/metrics.csv, metrics.json, overlay_<year>.png")
    print("overlay 顏色: 黃=訓練末年岸線, 綠=真實岸線, 紅=預測岸線")


# ----------------------------------------------------------------------------
# 岸線附近的「允許變動帶」: 推論時只讓模型改動這個區域, 其餘區域保留最後一張真實影像
# ----------------------------------------------------------------------------
def shore_band(frame_f, band_px=40, feather=8):
    """
    frame_f: HxWx3 in [-1,1]
    回傳 HxW float32 權重 (帶內=1, 往外 feather 像素內線性降到 0, 其餘=0)。
    只取「與影像上緣相連的主陸地」的海陸交界作為岸線;
    孤立在海中的突堤/礁石 (不與主陸地相連) 不會被納入, 因此一律原樣保留。
    """
    sea = water_mask(to_u8(frame_f))
    lab, n = ndi.label(~sea)
    top = [i for i in np.unique(lab[0, :]) if i != 0]
    main = np.isin(lab, top) if top else ~sea
    edge = main & ~ndi.binary_erosion(main, border_value=1)      # 影像邊框不算岸線
    if not edge.any():
        return np.zeros(sea.shape, np.float32)
    dist = ndi.distance_transform_edt(~edge)
    w = np.clip(1.0 - (dist - band_px) / max(feather, 1e-6), 0.0, 1.0)
    return w.astype(np.float32)


def make_protect(shape, boxes=None, mask_path=None, feather=1.5):
    """
    建立「強制保留」權重圖 (HxW, 1=完全沿用上一年影像, 不讓模型改動)。
    boxes: [(x0, y0, x1, y1), ...] 像素座標 (x 為橫向, y 為縱向, 左上為原點, 含頭不含尾)
    mask_path: 另一種指定方式, 白色(>127)的區域為保留區。
    feather: 邊緣以高斯模糊柔化, 避免出現接縫。
    """
    h, w = shape
    P = np.zeros((h, w), np.float32)
    for (x0, y0, x1, y1) in (boxes or []):
        P[max(int(y0), 0):min(int(y1), h), max(int(x0), 0):min(int(x1), w)] = 1.0
    if mask_path:
        m = Image.open(mask_path).convert("L")
        if (m.size[1], m.size[0]) != (h, w):
            m = m.resize((w, h), Image.NEAREST)
        P = np.maximum(P, (np.asarray(m) > 127).astype(np.float32))
    if feather and P.any():
        P = np.clip(ndi.gaussian_filter(P, feather) * 1.5, 0, 1)   # *1.5 讓區域內部維持 1
    return P


def smooth_mask(sea, sigma=3.0, min_area=300):
    """
    平滑二值陸/海遮罩 (消除岸線鋸齒、缺口、階梯狀凹凸與細碎斑點):
      高斯模糊後以 0.5 重新二值化 (比形態學開閉運算更能保持整體形狀), 再清理小碎塊。
    sigma 越大越平滑, 但岸線尖端與細小突出會被磨圓 (約移動 1~2 px);
    sigma<=0 代表不平滑。
    """
    sea = sea.astype(bool)
    if sigma <= 0:
        return sea
    p = int(3 * sigma) + 1
    x = ndi.gaussian_filter(np.pad(sea.astype(np.float32), p, mode="edge"), sigma)
    return clean_sea(x[p:-p, p:-p] > 0.5, min_area=min_area, smooth=False)