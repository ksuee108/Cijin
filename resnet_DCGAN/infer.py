"""
推論程式: 載入已訓練的 generator.pt, 做集成式自回歸預測 (不需重新訓練)
=====================================================================
需與訓練腳本 resnet_dcgan_coastal_v2.py 放在同一資料夾
(評估功能另需 coastal_metrics.py / gen_metrics.py)。

注意: k / ngf / n_resblocks 會自動由權重推斷, 不需指定 (若指定且與權重不符會報錯)。

範例:
  # 1) 預測未來 6 年 (2025-2036), 100 條軌跡
  python infer.py --data_dir "D:\desk\CECI\Coastline_CEC V\new\main file\data\north\Finish\segmentation\balanced\all\images" --ckpt ./final/generator.pt --out_dir ./infer_future --n_future 6 --mask_dir "D:\desk\CECI\Coastline_CEC V\new\main file\data\north\Finish\cover"

  # 2) 回測: 從 2021 出發預測 2022-2024, 並與真實影像評估
  #    (權重必須是「只用 2013-2021 訓練」的那一份, 否則是資料洩漏)
  python infer.py --data_dir "D:\desk\CECI\Coastline_CEC V\new\main file\data\north\Finish\segmentation\balanced\all\images" --ckpt ./val/generator.pt --out_dir ./infer_val --start_year 2021 --eval all --mask_dir "D:\desk\CECI\Coastline_CEC V\new\main file\data\north\Finish\cover"

"""
import argparse
import os
import types

import numpy as np
import torch
from PIL import Image

from resnet_dcgan_coastal_v2 import (CoastalSequenceDataset, ResnetGenerator,
                                     rollout, save_img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="存放 2013.jpg ... 2024.jpg 的資料夾")
    p.add_argument("--ckpt", required=True, help="generator.pt 路徑")
    p.add_argument("--out_dir", default="./infer_out")
    # 模型結構 (須與訓練一致)
    p.add_argument("--k", type=int, default=None, help="預設自動由權重推斷")
    p.add_argument("--ngf", type=int, default=None, help="預設自動由權重推斷")
    p.add_argument("--n_resblocks", type=int, default=None, help="預設自動由權重推斷")
    # 起點與長度
    p.add_argument("--start_year", type=int, default=None,
                   help="由哪一年的影像出發 (預設: 資料最後一年)")
    p.add_argument("--n_future", type=int, default=None,
                   help="預測幾年 (預設: 有真實影像就預測到最後一年, 否則 12)")
    # 集成設定
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--mode", choices=["step", "trajectory"], default="trajectory")
    p.add_argument("--agg", choices=["median", "mean"], default="median")
    p.add_argument("--noise_std", type=float, default=0.03)
    p.add_argument("--max_shift", type=int, default=2)
    p.add_argument("--infer_batch", type=int, default=10)
    p.add_argument("--band_px", type=int, default=20,
                   help="只允許岸線附近多少像素內變動 (0=關閉, 整張圖都由模型重畫)")
    p.add_argument("--band_feather", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # 評估 (僅在有真實影像可比對時有效)
    p.add_argument("--eval", choices=["none", "coast", "gen", "all"], default="none",
                   help="coast=岸線/陸水指標, gen=KID/FID/LPIPS/MS-SSIM, all=兩者")
    p.add_argument("--protect_box", action="append", default=[],
                   help="強制保留區 x0,y0,x1,y1 (像素, 可重複指定), 該區域一律沿用上一年影像")
    p.add_argument("--protect_mask", default=None, help="強制保留區遮罩圖 (白色=保留)")
    p.add_argument("--mask_dir", default=None,
                   help="歷年陸地遮罩資料夾 (檔名含年份, 例如 2013.png); 有的話作為真實岸線並校準分割")
    p.add_argument("--mask_polarity", choices=["auto", "land_white", "land_black"], default="auto",
                   help="遮罩中陸地的顏色; auto=依「陸在上、海在下」自動判斷")
    p.add_argument("--no_binary", action="store_true", help="不輸出二值化 (陸/海) 圖")
    p.add_argument("--binary_smooth", type=float, default=3.0,
                   help="二值圖平滑強度 sigma (像素), 消除岸線鋸齒/坑洞; 0=不平滑")
    p.add_argument("--binary_min_area", type=int, default=300, help="二值圖清除的小碎塊面積 (像素)")
    p.add_argument("--binary_polarity", choices=["land_white", "land_black"], default="land_white",
                   help="二值圖中陸地的顏色 (預設陸=白 255, 海=黑 0)")
    p.add_argument("--pixel_size_m", type=float, default=None, help="每像素公尺數 (岸線誤差換算)")
    p.add_argument("--fid_feature", type=int, default=2048)
    args = p.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    os.makedirs(args.out_dir, exist_ok=True)
 
    # 先讀權重, 由張量形狀推斷模型結構 (命令列有指定則以指定為準, 不符會提示)
    state = torch.load(args.ckpt, map_location="cpu")
    w0 = state["model.1.weight"]                       # [ngf, 3k, 7, 7]
    inf_ngf, inf_k = int(w0.shape[0]), int(w0.shape[1]) // 3
    inf_res = len({key.split(".")[1] for key in state if ".block." in key})
    for name, given, inferred in (("k", args.k, inf_k), ("ngf", args.ngf, inf_ngf),
                                  ("n_resblocks", args.n_resblocks, inf_res)):
        if given is not None and given != inferred:
            raise SystemExit(f"[error] --{name}={given} 與權重不符 (權重為 {inferred})")
    args.k, args.ngf, args.n_resblocks = inf_k, inf_ngf, inf_res
    print(f"[info] 由權重推斷: k={args.k}, ngf={args.ngf}, n_resblocks={args.n_resblocks}")
 
    # 由權重推斷模型結構 (第一層權重 shape = [ngf, 3*k, 7, 7])
    state = torch.load(args.ckpt, map_location=device)
    w0 = state["model.1.weight"]
    inf_ngf, inf_k = int(w0.shape[0]), int(w0.shape[1]) // 3
    inf_res = len({key.split(".")[1] for key in state if ".block." in key})
    for name, inferred in (("k", inf_k), ("ngf", inf_ngf), ("n_resblocks", inf_res)):
        given = getattr(args, name)
        if given is not None and given != inferred:
            print(f"[warn] --{name}={given} 與權重不符, 改用權重推斷的 {inferred}")
        setattr(args, name, inferred)
    print(f"[info] 由權重推斷: k={args.k}, ngf={args.ngf}, n_resblocks={args.n_resblocks}")
 
    # 載入影像 (holdout=0 → 不限制年份; 只用來讀圖)
    ds = CoastalSequenceDataset(args.data_dir, k=args.k, holdout=0)
    years = ds.years
    start = args.start_year if args.start_year is not None else years[-1]
    need = list(range(start - args.k + 1, start + 1))
    missing = [y for y in need if y not in ds.imgs]
    assert not missing, f"缺少起點所需的年份影像: {missing}"
    ds.train_end = start                      # 供評估函式當作「參考年」
    init = [ds.imgs[y] for y in need]
 
    if args.mask_dir:
        from coastal_metrics import setup_masks
        inv = {"auto": None, "land_white": False, "land_black": True}[args.mask_polarity]
        setup_masks(ds, args.mask_dir, start, inv)
 
    n_future = args.n_future
    if n_future is None:
        n_future = (years[-1] - start) if years[-1] > start else 12
    assert n_future > 0, "n_future 必須大於 0 (start_year 已是最後一年時請指定 --n_future)"
    truth_years = [start + i for i in range(1, n_future + 1) if (start + i) in ds.imgs]
    print(f"[info] 起點 {start}, 輸入年份 {need}, 預測 {start + 1}-{start + n_future}, "
          f"其中有真實影像可比對: {truth_years if truth_years else '無'}")
 
    # 載入模型
    residual = bool(int(state.get("residual_flag", torch.tensor(0))))   # 舊權重沒有此欄位 -> 非殘差
    state.setdefault("residual_flag", torch.tensor(0))
    print(f"[info] residual={residual}")
    G = ResnetGenerator(3 * args.k, 3, args.ngf, args.n_resblocks, residual=residual).to(device)
    try:
        G.load_state_dict(state)
    except RuntimeError as e:
        raise SystemExit(f"[error] 權重與模型結構不符, 請確認 --k/--ngf/--n_resblocks 與訓練一致\n{e}")
    G.eval()
 
    # 預測
    protect = None
    if args.protect_box or args.protect_mask:
        from coastal_metrics import make_protect
        boxes = [tuple(int(float(v)) for v in b.split(",")) for b in args.protect_box]
        protect = make_protect((ds.h, ds.w), boxes, args.protect_mask)
        print(f"[info] 強制保留區: {boxes if boxes else ''} {args.protect_mask or ''}")
    bands = []
    preds, stds = rollout(G, init, n_future, args.n_samples, args.mode,
                          args.noise_std, args.max_shift, args.infer_batch,
                          args.agg, device, args.seed,
                          band_px=args.band_px, band_feather=args.band_feather, bands=bands,
                          protect=protect)
 
    for i, (pr, sd) in enumerate(zip(preds, stds), 1):
        y = start + i
        save_img(pr, os.path.join(args.out_dir, f"pred_{y}.png"))
        s = (sd / (sd.max() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(s).save(os.path.join(args.out_dir, f"std_{y}.png"))
        
 
    # 評估
    if args.eval != "none":
        if not truth_years:
            print("[warn] 沒有對應年份的真實影像, 略過評估")
        else:
            ev = types.SimpleNamespace(out_dir=args.out_dir)
            if args.eval in ("coast", "all"):
                from coastal_metrics import validate_with_metrics
                validate_with_metrics(preds, ds, ev, px_m=args.pixel_size_m)
            if args.eval in ("gen", "all"):
                from gen_metrics import validate_gen_metrics
                validate_gen_metrics(preds, ds, ev, device=device, fid_feature=args.fid_feature)
 
 
if __name__ == "__main__":
    main()
 