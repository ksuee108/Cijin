import cv2
import numpy as np
import streamlit as st
import torch
import model.pidinet as pidinet_models

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ============================================================
# PiDiNet 邊緣偵測模型
# ============================================================

PIDINET_CKPT_PATH = r"model\pidinet.pth"

class PiDiNetArgs:
    model = "pidinet"
    config = "carv5"
    sa = False
    dil = False


@st.cache_resource
def load_pidinet_model():
    """
    比照 edge_detect.py 的模型建立 / 權重載入方式：
    getattr(models, "pidinet")(args)，去掉 state_dict 裡的
    "module." 前綴後 load_state_dict。
    """

    args = PiDiNetArgs()
    pidinet_model = getattr(pidinet_models, args.model)(args)

    checkpoint = torch.load(PIDINET_CKPT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    pidinet_model.load_state_dict(state_dict)

    pidinet_model = pidinet_model.to(device)
    pidinet_model.eval()

    return pidinet_model


def preprocess_for_pidinet(image_rgb_np):
    """
    比照 edge_detect.py 的 preprocess_patch：
    RGB -> BGR，減去 Caffe 風格的 channel mean，不做 0~1 normalize，
    轉成 CHW 的 torch tensor（batch 維度為 1）。
    """

    patch = image_rgb_np.astype(np.float32)
    patch = patch[:, :, ::-1]  # RGB -> BGR
    patch = patch - np.array([104.00699, 116.66877, 122.67892], dtype=np.float32)
    patch = patch.transpose(2, 0, 1)  # HWC -> CHW
    patch = np.ascontiguousarray(patch)

    return torch.from_numpy(patch).unsqueeze(0)


@torch.no_grad()
def run_pidinet_edge_detection(image_rgb_np):
    """
    輸入一張 RGB numpy 影像（這裡固定餵 segmentation 結果），
    回傳 0~255 的灰階邊緣機率圖，尺寸與輸入相同。
    """

    pidinet_model = load_pidinet_model()

    h, w = image_rgb_np.shape[:2]

    input_tensor = preprocess_for_pidinet(image_rgb_np).to(device)

    outputs = pidinet_model(input_tensor)

    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    edge_map = (edge * 255).astype(np.uint8)

    edge_map = cv2.resize(edge_map, (w, h), interpolation=cv2.INTER_LINEAR)

    return edge_map


def robust_binarize_edge(edge_map):
    """
    比照 edge_detect.py 的 robust_binarize：
    高斯模糊 + Otsu 二值化，再骨架化（skeletonize）把粗邊緣帶
    收縮成 1 像素寬的中心線。
    """

    from skimage.morphology import skeletonize

    edge_blur = cv2.GaussianBlur(edge_map, (5, 5), 0)

    _, binary = cv2.threshold(
        edge_blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    skeleton = skeletonize(binary > 0)
    binary = (skeleton.astype(np.uint8)) * 255

    return binary


# ============================================================
# DiffusionEdge 邊緣偵測（以 command line 執行 demo.py）
# ============================================================

DIFFUSIONEDGE_DIR = "./model/diffusionedge"
DIFFUSIONEDGE_PRE_WEIGHT = "./model-4.pt"
DIFFUSIONEDGE_BATCH_SIZE = 1


def run_diffusionedge_edge_detection(image_rgb_np):
    """

    做法：
      1. 建立暫存的輸入 / 輸出資料夾，把輸入影像存成一張圖片放進輸入資料夾。
      2. 用 subprocess 呼叫 demo.py（cwd 設成 DIFFUSIONEDGE_DIR，
         這樣 --pre_weight 的相對路徑才會對）。
      3. demo.py 跑完後，到輸出資料夾（含子資料夾）裡找結果圖片讀回來，
         轉成灰階、resize 回輸入尺寸後回傳，跟 PiDiNet 的輸出格式一致。

    註：demo.py 實際把結果存成什麼檔名、存在 out_dir 的哪一層，
    目前是用「找 out_dir 底下第一張圖片」來處理；如果實際結構不是這樣
    （例如檔名被改成別的、或分成好幾個子資料夾), 請告訴我 demo.py
    的輸出邏輯，我再把尋找結果檔案的這段改精確。
    """

    input_dir = tempfile.mkdtemp(prefix="diffusionedge_in_")
    out_dir = tempfile.mkdtemp(prefix="diffusionedge_out_")

    try:
        input_filename = "input.png"
        input_path = os.path.join(input_dir, input_filename)

        Image.fromarray(image_rgb_np).save(input_path)

        command = [
            sys.executable,
            "demo.py",
            "--input_dir", input_dir,
            "--pre_weight", DIFFUSIONEDGE_PRE_WEIGHT,
            "--out_dir", out_dir,
            "--bs", str(DIFFUSIONEDGE_BATCH_SIZE)
        ]

        result = subprocess.run(
            command,
            cwd=DIFFUSIONEDGE_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "DiffusionEdge (demo.py) 執行失敗（returncode="
                f"{result.returncode}）：\n\n"
                f"--- stdout ---\n{result.stdout}\n\n"
                f"--- stderr ---\n{result.stderr}"
            )

        output_candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            output_candidates.extend(
                glob.glob(os.path.join(out_dir, "**", ext), recursive=True)
            )

        if len(output_candidates) == 0:
            raise RuntimeError(
                "DiffusionEdge 執行完成，但在輸出資料夾裡找不到任何結果圖，"
                f"請確認 demo.py 實際把結果存到哪裡（目前查找路徑：{out_dir}）。\n\n"
                f"--- stdout ---\n{result.stdout}"
            )

        # 優先找檔名（不含副檔名）跟輸入檔名一樣的結果，找不到就取第一張
        input_stem = os.path.splitext(input_filename)[0]
        matched = [
            p for p in output_candidates
            if os.path.splitext(os.path.basename(p))[0] == input_stem
        ]
        output_path = matched[0] if matched else sorted(output_candidates)[0]

        result_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)

        if result_image is None:
            raise RuntimeError(f"讀取 DiffusionEdge 輸出圖片失敗：{output_path}")

        if result_image.ndim == 3:
            edge_map = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        else:
            edge_map = result_image

        h, w = image_rgb_np.shape[:2]

        if edge_map.shape[:2] != (h, w):
            edge_map = cv2.resize(edge_map, (w, h), interpolation=cv2.INTER_LINEAR)

        if edge_map.dtype != np.uint8:
            edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return edge_map

    finally:
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)