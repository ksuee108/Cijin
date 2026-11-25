"""
推論程式: 載入已訓練的 generator.pt, 做集成式自回歸預測 (不需重新訓練)
=====================================================================
使用 resnet_dcgan_cijin.py (多地點版資料集)。需與下列檔案放在同一資料夾:
    resnet_dcgan_cijin.py              (必要)
    coastal_metrics.py / gen_metrics.py (只有用到 --eval / --mask_dir / --band_px / --protect_* 才需要)

資料夾結構 (兩種都可以):
    data_dir/2013.jpg ... 2024.jpg            -> 單一地點
    data_dir/海水/2013.jpg ...  data_dir/風車/2013.jpg ...  -> 多地點 (用 --sites 可只跑部分)

輸出位置:
    只有一個地點 -> out_dir/pred_YYYY.png
    多個地點     -> out_dir/<地點>/pred_YYYY.png

注意:
  * k / ngf / n_resblocks / residual 由權重自動推斷; color_match 預設讀 generator.pt 旁邊的 config.json。
  * 權重必須是用 resnet_dcgan_cijin.py 訓練的 (舊的 v2 權重沒有色彩校正, 結果會不準)。

範例:
  # 1) 預測未來 6 年
  python infer.py --data_dir ./cijin --ckpt ./future/generator.pt --out_dir ./infer_future --n_future 6

  # 2) 回測: 從 2023 出發預測 2024-2026, 與真實影像評估
  #    (權重必須是 --holdout 3 訓練出來的那一份, 否則是資料洩漏)
  python infer.py --data_dir ./cijin --ckpt ./val/generator.pt --out_dir ./infer_val --start_year 2023 --eval all
"""
import argparse
import json
import os
import types

import numpy as np
import torch
from PIL import Image

from resnet_dcgan_cijin import (CoastalSequenceDataset, ResnetGenerator,
                                rollout, save_img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="存放 2013.jpg ... 的資料夾 (或其子資料夾 = 地點)")
    p.add_argument("--ckpt", required=True, help="generator.pt 路徑")
    p.add_argument("--out_dir", default="./infer_out")
    p.add_argument("--sites", nargs="*", default=None, help="只跑指定地點")
    # 模型結構 (須與訓練一致, 預設自動推斷)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--ngf", type=int, default=None)
    p.add_argument("--n_resblocks", type=int, default=None)
    p.add_argument("--color_match", choices=["auto", "meanstd", "none"], default="auto",
                   help="auto=讀 ckpt 旁邊的 config.json, 找不到就用 meanstd (需與訓練一致)")
    # 起點與長度
    p.add_argument("--start_year", type=int, default=None,
                   help="由哪一年的影像出發 (預設: 各地點的最後一年)")
    p.add_argument("--n_future", type=int, default=None,
                   help="預測幾年 (預設: 有真實影像就預測到最後一年, 否則 12)")
    # 集成設定 (1024x768 整張推論很吃記憶體, 預設值已調小)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--mode", choices=["step", "trajectory"], default="trajectory")
    p.add_argument("--agg", choices=["median", "mean"], default="median")
    p.add_argument("--noise_std", type=float, default=0.03)
    p.add_argument("--max_shift", type=int, default=2)
    p.add_argument("--infer_batch", type=int, default=2)
    p.add_argument("--band_px", type=int, default=0,
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
                   help="歷年陸地遮罩資料夾 (檔名含年份); 多地點時先找 mask_dir/<地點>/, 找不到就用 mask_dir")
    p.add_argument("--mask_polarity", choices=["auto", "land_white", "land_black"], default="auto")
    p.add_argument("--pixel_size_m", type=float, default=None, help="每像素公尺數 (岸線誤差換算)")
    p.add_argument("--fid_feature", type=int, default=2048)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 讀權重, 由張量形狀推斷模型結構 ----
    state = torch.load(args.ckpt, map_location="cpu")
    w0 = state["model.1.weight"]                       # [ngf, 3k, 7, 7]
    inferred = {"ngf": int(w0.shape[0]), "k": int(w0.shape[1]) // 3,
                "n_resblocks": len({key.split(".")[1] for key in state if ".block." in key})}
    for name, val in inferred.items():
        given = getattr(args, name)
        if given is not None and given != val:
            raise SystemExit(f"[error] --{name}={given} 與權重不符 (權重為 {val})")
        setattr(args, name, val)
    residual = bool(int(state.get("residual_flag", torch.tensor(0))))   # 舊權重沒有此欄位 -> 非殘差
    state.setdefault("residual_flag", torch.tensor(0))
    print(f"[info] 由權重推斷: k={args.k}, ngf={args.ngf}, n_resblocks={args.n_resblocks}, residual={residual}")

    G = ResnetGenerator(3 * args.k, 3, args.ngf, args.n_resblocks, residual=residual).to(device)
    try:
        G.load_state_dict(state)
    except RuntimeError as e:
        raise SystemExit(f"[error] 權重與模型結構不符\n{e}")
    G.eval()

    # ---- 色彩校正模式要與訓練一致 ----
    color_match = args.color_match
    if color_match == "auto":
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.json")
        color_match = "meanstd"
        if os.path.exists(cfg_path):
            raw = open(cfg_path, "rb").read()
            for enc in ("utf-8-sig", "cp950", "gbk"):
                try:
                    color_match = json.loads(raw.decode(enc)).get("color_match", "meanstd")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        else:
            print("[warn] 找不到 config.json, color_match 預設 meanstd")
    print(f"[info] color_match={color_match}")

    # ---- 載入影像 (holdout=0 → 不限制年份; 只用來讀圖與色彩校正) ----
    ds = CoastalSequenceDataset(args.data_dir, sites=args.sites, k=args.k,
                                holdout=0, color_match=color_match)
    single = len(ds.sites) == 1

    boxes = [tuple(int(float(v)) for v in b.split(",")) for b in args.protect_box]
    done = 0
    for name, site in ds.sites.items():
        imgs, years = site["imgs"], site["years"]
        site_dir = args.out_dir if single else os.path.join(args.out_dir, name)
        os.makedirs(site_dir, exist_ok=True)
        print(f"\n===== 地點: {name} =====")

        start = args.start_year if args.start_year is not None else years[-1]
        need = list(range(start - args.k + 1, start + 1))
        missing = [y for y in need if y not in imgs]
        if missing:
            msg = f"[{name}] 缺少起點所需的連續年份影像: {missing}"
            if single:
                raise SystemExit("[error] " + msg)
            print("[warn] " + msg + ", 略過此地點")
            continue
        init = [imgs[y] for y in need]

        n_future = args.n_future
        if n_future is None:
            n_future = (years[-1] - start) if years[-1] > start else 12
        if n_future <= 0:
            raise SystemExit("[error] n_future 必須大於 0 (start_year 已是最後一年時請指定 --n_future)")
        truth_years = [start + i for i in range(1, n_future + 1) if (start + i) in imgs]
        print(f"[info] 起點 {start}, 輸入年份 {need}, 預測 {start + 1}-{start + n_future}, "
              f"可比對的真實年份: {truth_years if truth_years else '無'}")

        # 評估模組需要的「資料集介面」: imgs / years / h / w / train_end (/ sea_masks)
        view = types.SimpleNamespace(imgs=imgs, years=years, h=site["h"], w=site["w"], train_end=start)

        if args.mask_dir:
            from coastal_metrics import setup_masks
            sub = os.path.join(args.mask_dir, name)
            mdir = sub if os.path.isdir(sub) else args.mask_dir
            inv = {"auto": None, "land_white": False, "land_black": True}[args.mask_polarity]
            setup_masks(view, mdir, start, inv)

        protect = None
        if boxes or args.protect_mask:
            from coastal_metrics import make_protect
            protect = make_protect((site["h"], site["w"]), boxes, args.protect_mask)
            print(f"[info] 強制保留區: {boxes if boxes else ''} {args.protect_mask or ''}")

        bands = []
        preds, stds = rollout(G, init, n_future, args.n_samples, args.mode,
                              args.noise_std, args.max_shift, args.infer_batch,
                              args.agg, device, args.seed,
                              band_px=args.band_px, band_feather=args.band_feather,
                              bands=bands, protect=protect)

        for i, (pr, sd) in enumerate(zip(preds, stds), 1):
            y = start + i
            save_img(pr, os.path.join(site_dir, f"pred_{y}.png"))
            s = (sd / (sd.max() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(s).save(os.path.join(site_dir, f"std_{y}.png"))
            if bands:
                Image.fromarray((bands[i - 1] * 255).astype(np.uint8)).save(
                    os.path.join(site_dir, f"band_{y}.png"))
        done += 1

        # ---- 評估 ----
        if args.eval != "none":
            if not truth_years:
                print("[warn] 沒有對應年份的真實影像, 略過評估")
            else:
                ev = types.SimpleNamespace(out_dir=site_dir)
                if args.eval in ("coast", "all"):
                    from coastal_metrics import validate_with_metrics
                    validate_with_metrics(preds, view, ev, px_m=args.pixel_size_m)
                if args.eval in ("gen", "all"):
                    from gen_metrics import validate_gen_metrics
                    validate_gen_metrics(preds, view, ev, device=device, fid_feature=args.fid_feature)

    if done == 0:
        raise SystemExit("[error] 沒有任何地點完成預測")


if __name__ == "__main__":
    main()