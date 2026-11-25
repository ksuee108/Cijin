import io
import numpy as np
import streamlit as st
import torch
import albumentations as A

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from config import ALL_CLASSES, LABEL_COLORS_LIST

import re
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path
import time
from collections import Counter
from datetime import datetime
from histogram_match import match_histogram_to_reference, find_reference_images
from Semantic_Segmentation import predict_roi, decode_segmap, create_overlay, apply_correction_objects, parse_rect_bbox_from_canvas, \
                                    load_model, class_legend_markdown
from edge_detection import run_pidinet_edge_detection, run_diffusionedge_edge_detection
# ============================================================
# 基本設定
# ============================================================

st.set_page_config(
    page_title="台灣世曦 港灣部 岸線偵測系統",
    layout="wide"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# Session State
# ============================================================

DEFAULT_SESSION_STATE = {
    "uploaded_image": None,
    "roi": None,
    "prediction": None,          # 目前(可能已修正)的完整遮罩
    "original_prediction": None,  # AI 剛推論完、尚未人工修正的遮罩
    "mask_history": [],          # 修正歷史紀錄，用於復原
    "overlay": None,
    "segmentation": None,
    "canvas_key": 0,             # ROI 選取畫布的 key
    "correction_canvas_key": 0,  # 人工修正畫布的 key
    "cropped_image": None,       # 剪裁後的影像（PIL Image），None 代表未剪裁
    "crop_canvas_key": 0,        # 剪裁畫布的 key
    "roi_base_image": None,      # AI 分割時裁下來的 ROI 原始彩色影像（PIL Image）
    "edge_map": None,            # PiDiNet / DiffusionEdge 邊緣機率圖（灰階）
    "edge_skeleton": None,       # 二值化骨架線
    "edge_model_used": None,     # 這次邊緣偵測用的是哪個模型
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============================================================
# Title
# ============================================================

st.title("台灣世曦 港灣部 岸線偵測與預測系統")
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    ["岸線偵測", "岸線變遷分析", "新增衛星影像及重新訓練", "關於本系統"]
)

with main_tab1:
    st.header("岸線偵測模組")
    st.markdown(
        """
    本系統採用 **DeepLabV3-ResNet101** 進行語意分割，並支援 **AI 輔助人工修正**
    （操作方式類似 [CVAT](https://www.cvat.ai/) 的語意分割標註工具）。

    操作方式：

    1. 上傳影像
    2.（選用）拖曳矩形框剪裁影像，去除不需要的範圍
    3. 在影像上以滑鼠拖曳建立矩形 ROI
    4. 按下「開始語意分割」，AI 模型僅針對 ROI 進行推論
    5. 在「人工修正」區塊選擇類別，使用多邊形或筆刷工具修正 AI 分割結果
    6. 下載修正後的 segmentation mask / overlay / 類別索引遮罩
    """
    )


    # ============================================================
    # Image Upload
    # ============================================================

    uploaded_file = st.file_uploader(
        "請上傳影像",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=False
    )


    # ============================================================
    # Main
    # ============================================================

    if uploaded_file is not None:

        # --------------------------------------------------------
        # Load & adjust image
        # --------------------------------------------------------

        original_image = Image.open(uploaded_file).convert("RGB")

        full_width, full_height = original_image.size

        MAX_CANVAS_WIDTH = 1200
        MAX_CANVAS_HEIGHT = 700

        crop_scale = min(
            MAX_CANVAS_WIDTH / full_width,
            MAX_CANVAS_HEIGHT / full_height,
            1.0
        )

        crop_canvas_width = int(full_width * crop_scale)
        crop_canvas_height = int(full_height * crop_scale)

        # ========================================================
        # ① 剪裁影像（選用）
        # ========================================================

        st.subheader("① 剪裁影像（選用）")

        st.info(
            "如需先去除影像中不需要的邊緣或範圍，可在下方拖曳矩形框選擇要保留的區域，"
            "按「套用剪裁」後，後續的 ROI 選取與語意分割都會以剪裁後的影像為準。"
            "若不需要剪裁，直接略過此步驟即可。"
        )

        crop_background = (
            st.session_state.cropped_image
            if st.session_state.cropped_image is not None
            else original_image
        )

        crop_canvas_result = st_canvas(
            fill_color="rgba(0, 200, 0, 0.15)",
            stroke_width=3,
            stroke_color="#00C800",
            background_image=crop_background,
            update_streamlit=True,
            height=crop_canvas_height,
            width=crop_canvas_width,
            drawing_mode="rect",
            point_display_radius=0,
            key=f"crop_canvas_{st.session_state.crop_canvas_key}"
        )

        crop_apply_col, crop_reset_col, crop_info_col = st.columns([1, 1, 2])

        with crop_apply_col:
            crop_apply_button = st.button(
                "套用剪裁", use_container_width=True
            )

        with crop_reset_col:
            crop_reset_button = st.button(
                "清除剪裁", use_container_width=True
            )

        if crop_apply_button:

            crop_source = (
                st.session_state.cropped_image
                if st.session_state.cropped_image is not None
                else original_image
            )
            crop_src_width, crop_src_height = crop_source.size

            crop_bbox = parse_rect_bbox_from_canvas(
                crop_canvas_result,
                crop_src_width,
                crop_src_height,
                crop_canvas_width,
                crop_canvas_height
            )

            if crop_bbox is None:
                st.warning("請先在影像上拖曳矩形框，選擇要保留的剪裁範圍。")

            else:
                st.session_state.cropped_image = crop_source.crop(crop_bbox)

                # 影像尺寸已改變，先前的 ROI / AI 結果不再對應，一併清除
                st.session_state.roi = None
                st.session_state.prediction = None
                st.session_state.original_prediction = None
                st.session_state.mask_history = []
                st.session_state.overlay = None
                st.session_state.segmentation = None
                st.session_state.roi_base_image = None
                st.session_state.edge_map = None
                st.session_state.edge_skeleton = None
                st.session_state.edge_model_used = None
                st.session_state.canvas_key += 1
                st.session_state.correction_canvas_key += 1
                st.session_state.crop_canvas_key += 1

                st.rerun()

        if crop_reset_button and st.session_state.cropped_image is not None:

            st.session_state.cropped_image = None

            st.session_state.roi = None
            st.session_state.prediction = None
            st.session_state.original_prediction = None
            st.session_state.mask_history = []
            st.session_state.overlay = None
            st.session_state.segmentation = None
            st.session_state.roi_base_image = None
            st.session_state.edge_map = None
            st.session_state.edge_skeleton = None
            st.session_state.edge_model_used = None
            st.session_state.canvas_key += 1
            st.session_state.correction_canvas_key += 1
            st.session_state.crop_canvas_key += 1

            st.rerun()

        if st.session_state.cropped_image is not None:
            st.success(
                f"目前已套用剪裁，影像尺寸：{st.session_state.cropped_image.size[0]} × "
                f"{st.session_state.cropped_image.size[1]}"
            )

        # --------------------------------------------------------
        # working_image：後續 ROI 選取、AI 推論、人工修正、下載
        # 都以這張（可能已剪裁的）影像為準
        # --------------------------------------------------------

        working_image = (
            st.session_state.cropped_image
            if st.session_state.cropped_image is not None
            else original_image
        )

        width, height = working_image.size

        scale = min(
            MAX_CANVAS_WIDTH / width,
            MAX_CANVAS_HEIGHT / height,
            1.0
        )

        canvas_width = int(width * scale)
        canvas_height = int(height * scale)

        # 畫布座標 → 原始影像座標 的縮放係數
        scale_x = width / float(canvas_width)
        scale_y = height / float(canvas_height)

        # ========================================================
        # ② 選擇語意分割區域 (ROI)
        # ========================================================

        st.markdown("---")
        st.subheader("② 選擇語意分割區域")

        st.info("請使用滑鼠在影像上拖曳矩形框，選擇需要進行 AI 語意分割的區域。")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.15)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=working_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            point_display_radius=0,
            key=f"canvas_{st.session_state.canvas_key}"
        )

        current_roi = parse_rect_bbox_from_canvas(
            canvas_result, width, height, canvas_width, canvas_height
        )

        if current_roi is not None:
            st.session_state.roi = current_roi

        display_roi = current_roi if current_roi is not None else st.session_state.get("roi", None)

        if display_roi is not None:

            x1, y1, x2, y2 = display_roi
            roi_width = x2 - x1
            roi_height = y2 - y1

            st.success(
                f"已選擇 ROI  X: {x1} ~ {x2}  |  Y: {y1} ~ {y2}  |  "
                f"ROI 尺寸: {roi_width} × {roi_height}"
            )

        else:
            st.warning("尚未選擇 ROI，請在影像上拖曳矩形框。")

        
        # ---- 直方圖匹配（以 2024 年影像為參考）----
        ref_candidates = find_reference_images(Path(__file__).resolve().parent / "cijin", 2024)

        use_hist_match = st.checkbox(
            "分割前先用 2024 年影像做直方圖匹配（校正色調差異）",
            value=True,
            key="use_hist_match",
            disabled=len(ref_candidates) == 0,
        )

        hist_ref_path = None
        if ref_candidates:
            ref_label = st.selectbox(
                "2024 年參考影像", list(ref_candidates), key="hist_ref_choice",
                disabled=not use_hist_match,
            )
            hist_ref_path = ref_candidates[ref_label]
        else:
            st.warning("在 cijin 資料集中找不到 2024 年影像，已略過直方圖匹配。")
            use_hist_match = False


        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            detect_button = st.button("開始語意分割", type="primary", use_container_width=True)

        with col2:
            clear_button = st.button("清除 ROI", use_container_width=True)

        if clear_button:
            st.session_state.roi = None
            st.session_state.prediction = None
            st.session_state.original_prediction = None
            st.session_state.mask_history = []
            st.session_state.overlay = None
            st.session_state.segmentation = None
            st.session_state.roi_base_image = None
            st.session_state.edge_map = None
            st.session_state.edge_skeleton = None
            st.session_state.edge_model_used = None
            st.session_state.canvas_key += 1
            st.session_state.correction_canvas_key += 1
            st.rerun()

        model = load_model()
        # ========================================================
        # AI 語意分割
        # ========================================================

        if detect_button:

            roi = st.session_state.roi

            if roi is None:
                st.error("請先在影像上框選 ROI。")

            else:

                x1, y1, x2, y2 = roi

                image_np = np.array(working_image)

                if use_hist_match and hist_ref_path is not None:
                    with Image.open(hist_ref_path) as ref_img:
                        matched_np = match_histogram_to_reference(working_image, ref_img)
                else:
                    matched_np = image_np

                roi_np = image_np[y1:y2, x1:x2]                 # 原色，用於疊圖顯示
                roi_matched_np = matched_np[y1:y2, x1:x2]       # 匹配後，只給模型推論
                roi_image = Image.fromarray(roi_np)

                progress_text = st.empty()
                progress_text.info("DeepLabV3-ResNet101 正在進行 ROI 語意分割...")

                prediction_roi = predict_roi(Image.fromarray(roi_matched_np), model, patch_size=512)

                # 不再把預測結果擴回整張影像，直接保留 ROI 裁切後的大小，
                # ROI 以外的區域整個捨棄。
                segmentation_rgb = decode_segmap(prediction_roi, LABEL_COLORS_LIST)
                overlay = create_overlay(roi_image, prediction_roi, roi=None)

                st.session_state.prediction = prediction_roi
                st.session_state.original_prediction = prediction_roi.copy()
                st.session_state.mask_history = [prediction_roi.copy()]
                st.session_state.segmentation = segmentation_rgb
                st.session_state.overlay = overlay
                st.session_state.roi_base_image = roi_image
                st.session_state.correction_canvas_key += 1

                progress_text.success("ROI 語意分割完成，可於下方進行人工修正。")

        # ========================================================
        # ② AI 分割結果
        # ========================================================

        if st.session_state.overlay is not None:

            st.markdown("---")
            st.subheader("③ Semantic Segmentation 結果")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.image(
                    st.session_state.segmentation,
                    caption="Semantic Segmentation",
                    use_container_width=False
                )

            with result_col2:
                st.image(
                    st.session_state.overlay,
                    caption="Original + Segmentation Overlay",
                    use_container_width=False
                )

            # ====================================================
            # ③ AI 輔助人工修正（類似 CVAT）
            # ====================================================

            st.markdown("---")
            st.subheader("④ 人工修正（多邊形 / 筆刷，類似 CVAT AI-assisted annotation）")

            st.markdown(
                class_legend_markdown(ALL_CLASSES, LABEL_COLORS_LIST),
                unsafe_allow_html=True
            )

            st.info(
                "選擇要標註的類別與工具後，在下方影像上畫出要修正的區域，"
                "按「套用修正」即會寫入遮罩；若要清除某區域，選擇背景類別（index 0）畫過去即可。"
            )
            st.info(
                "若選擇多邊形或套索工具，請在影像上點擊建立多邊形輪廓，最後點擊起點完成封閉(圓點消失才算完成)；"
                "若選擇筆刷工具，請在影像上拖曳塗抹。")

            correction_col1, correction_col2, correction_col3 = st.columns([2, 2, 1])

            with correction_col1:
                class_options = list(enumerate(ALL_CLASSES))
                selected_class_index = st.selectbox(
                    "標註類別",
                    options=[idx for idx, _ in class_options],
                    format_func=lambda idx: f"{idx}: {ALL_CLASSES[idx]}",
                    key="selected_class_index"
                )

            with correction_col2:
                tool_mode_label = st.radio(
                    "修正工具",
                    options=["多邊形（填滿區域）", "筆刷（塗抹）"],
                    horizontal=True,
                    key="tool_mode_label"
                )
                tool_mode = "polygon" if tool_mode_label.startswith("多邊形") else "brush"

            with correction_col3:
                brush_size = st.slider(
                    "筆刷大小",
                    min_value=3,
                    max_value=80,
                    value=20,
                    disabled=(tool_mode != "brush")
                )

            correction_drawing_mode = "polygon" if tool_mode == "polygon" else "freedraw"
            selected_color = LABEL_COLORS_LIST[selected_class_index]
            selected_color_css = f"rgba({selected_color[0]},{selected_color[1]},{selected_color[2]},0.6)"

            correction_background = Image.fromarray(st.session_state.overlay)

            # 修正畫布是以「裁切後的 ROI 影像」為準，尺寸與 step② 選 ROI 用的
            # 整張影像不同，所以另外計算一組縮放比例。
            corr_width, corr_height = correction_background.size

            corr_scale = min(
                MAX_CANVAS_WIDTH / corr_width,
                MAX_CANVAS_HEIGHT / corr_height,
                1.0
            )

            corr_canvas_width = int(corr_width * corr_scale)
            corr_canvas_height = int(corr_height * corr_scale)

            corr_scale_x = corr_width / float(corr_canvas_width)
            corr_scale_y = corr_height / float(corr_canvas_height)

            correction_canvas_result = st_canvas(
                fill_color=selected_color_css,
                stroke_width=brush_size if tool_mode == "brush" else 2,
                stroke_color=selected_color_css,
                background_image=correction_background,
                update_streamlit=True,
                height=corr_canvas_height,
                width=corr_canvas_width,
                drawing_mode=correction_drawing_mode,
                point_display_radius=0,
                key=f"correction_canvas_{st.session_state.correction_canvas_key}"
            )

            n_shapes = 0
            if correction_canvas_result.json_data is not None:
                n_shapes = len(correction_canvas_result.json_data.get("objects", []))

            apply_col, undo_col, reset_col = st.columns([1, 1, 1])

            with apply_col:
                apply_button = st.button(
                    "套用修正", type="primary", use_container_width=True
                )

            with undo_col:
                undo_button = st.button(
                    "復原上一步", use_container_width=True,
                    disabled=(len(st.session_state.mask_history) <= 1)
                )

            with reset_col:
                reset_button = st.button(
                    "重設回 AI 原始結果", use_container_width=True
                )

            if apply_button:

                objects = []

                if correction_canvas_result.json_data is not None:
                    objects = correction_canvas_result.json_data.get("objects", [])

                if len(objects) == 0:
                    st.warning("尚未在影像上畫出任何修正區域。")

                else:

                    updated_mask = apply_correction_objects(
                        st.session_state.prediction,
                        objects,
                        corr_scale_x,
                        corr_scale_y,
                        tool_mode,
                        selected_class_index,
                        brush_size
                    )

                    st.session_state.mask_history.append(updated_mask.copy())
                    st.session_state.prediction = updated_mask

                    st.session_state.segmentation = decode_segmap(
                        updated_mask, LABEL_COLORS_LIST
                    )
                    st.session_state.overlay = create_overlay(
                        st.session_state.roi_base_image, updated_mask, roi=None
                    )

                    st.session_state.correction_canvas_key += 1

                    st.rerun()

            if undo_button and len(st.session_state.mask_history) > 1:

                st.session_state.mask_history.pop()
                restored_mask = st.session_state.mask_history[-1].copy()

                st.session_state.prediction = restored_mask
                st.session_state.segmentation = decode_segmap(restored_mask, LABEL_COLORS_LIST)
                st.session_state.overlay = create_overlay(
                    st.session_state.roi_base_image, restored_mask, roi=None
                )

                st.session_state.correction_canvas_key += 1

                st.rerun()

            if reset_button and st.session_state.original_prediction is not None:

                restored_mask = st.session_state.original_prediction.copy()

                st.session_state.prediction = restored_mask
                st.session_state.mask_history = [restored_mask.copy()]
                st.session_state.segmentation = decode_segmap(restored_mask, LABEL_COLORS_LIST)
                st.session_state.overlay = create_overlay(
                    st.session_state.roi_base_image, restored_mask, roi=None
                )

                st.session_state.correction_canvas_key += 1

                st.rerun()

            # ====================================================
            # ⑤ 邊緣偵測（PiDiNet / DiffusionEdge）
            # ====================================================

            st.markdown("---")
            st.subheader("⑤ 邊緣偵測")

            st.info(
                "以目前的 Semantic Segmentation 結果（含人工修正）作為輸入，"
                "偵測分割邊界的邊緣線。"
            )

            edge_col1, edge_col2 = st.columns([2, 1])

            with edge_col1:
                edge_model_choice = st.radio(
                    "邊緣偵測模型",
                    options=["PiDiNet", "DiffusionEdge"],
                    horizontal=True,
                    key="edge_model_choice"
                )

            with edge_col2:
                run_edge_button = st.button(
                    "執行邊緣偵測", use_container_width=True
                )

            if run_edge_button:

                if edge_model_choice == "PiDiNet":

                    edge_progress = st.empty()
                    edge_progress.info("PiDiNet 正在偵測邊緣...")

                    edge_map = run_pidinet_edge_detection(st.session_state.segmentation)

                    st.session_state.edge_map = edge_map

                    st.session_state.edge_model_used = "PiDiNet"

                    edge_progress.success("PiDiNet 邊緣偵測完成。")

                else:

                    edge_progress = st.empty()
                    edge_progress.info(
                        "DiffusionEdge 正在偵測邊緣，這是擴散模型，推論會比 PiDiNet"
                        "慢上不少，請耐心等候（畫面看起來像卡住是正常的）..."
                    )

                    try:
                        edge_map = run_diffusionedge_edge_detection(
                            st.session_state.segmentation
                        )

                        st.session_state.edge_map = edge_map
                        st.session_state.edge_model_used = "DiffusionEdge"

                        edge_progress.success("DiffusionEdge 邊緣偵測完成。")

                    except Exception as exc:
                        edge_progress.empty()
                        st.error(f"DiffusionEdge 執行失敗：{exc}")

            if st.session_state.edge_map is not None:


                st.image(
                    st.session_state.edge_map,
                    caption=f"邊緣機率圖（{st.session_state.edge_model_used}）",
                    use_container_width=False
                )


            # ====================================================
            # ⑥ 下載結果
            # ====================================================

            st.markdown("---")
            st.markdown("### 下載結果")

            segmentation_pil = Image.fromarray(st.session_state.segmentation)
            segmentation_buffer = io.BytesIO()
            segmentation_pil.save(segmentation_buffer, format="PNG")

            overlay_pil = Image.fromarray(st.session_state.overlay)
            overlay_buffer = io.BytesIO()
            overlay_pil.save(overlay_buffer, format="PNG")

            indexed_mask_pil = Image.fromarray(st.session_state.prediction)
            indexed_mask_buffer = io.BytesIO()
            indexed_mask_pil.save(indexed_mask_buffer, format="PNG")

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                st.download_button(
                    label="下載 Segmentation Mask",
                    data=segmentation_buffer.getvalue(),
                    file_name="segmentation.png",
                    mime="image/png",
                    use_container_width=True
                )

            with download_col2:
                st.download_button(
                    label="下載 Overlay",
                    data=overlay_buffer.getvalue(),
                    file_name="overlay.png",
                    mime="image/png",
                    use_container_width=True
                )

            if st.session_state.edge_map is not None:

                edge_map_pil = Image.fromarray(st.session_state.edge_map)
                edge_map_buffer = io.BytesIO()
                edge_map_pil.save(edge_map_buffer, format="PNG")

                st.download_button(
                    label="下載邊緣機率圖",
                    data=edge_map_buffer.getvalue(),
                    file_name="edge_map.png",
                    mime="image/png",
                    use_container_width=True
                )


BASE_DIR = Path(__file__).resolve().parent
INFER_SCRIPT = BASE_DIR / "infer.py"
CKPT_PATH = BASE_DIR / "test" / "generator.pt"
OUT_DIR = BASE_DIR / "output"

with main_tab2:
    st.header("岸線變遷分析模組")
    st.markdown(
        "上傳*多張*不同年份的影像（檔名需為年份，例如 `2013.png`），"
        "系統會以已訓練的生成模型預測未來年份的影像。"
    )

    uploaded_files = st.file_uploader(
        "請上傳多年份影像",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="change_uploader",
    )

    valid_files = []
    for file in uploaded_files or []:
        if re.fullmatch(r"(19|20)\d{2}\.(jpg|jpeg|png|bmp|tif|tiff)", file.name, re.IGNORECASE):
            valid_files.append(file)
        else:
            st.error(f"檔名格式錯誤：{file.name}，請使用例如 2013.png 或 2024.jpg 的格式。")

    ckpt_input = st.text_input(
        "模型權重檔路徑（generator.pt）\
        如有多個版本，請使用最新的權重檔",
        value=str(CKPT_PATH),
        key="ckpt_path_input",
        help="可直接貼上完整路徑；檔案總管「複製為路徑」帶的引號會自動去掉。",
    )
    ckpt_path = Path(ckpt_input.strip().strip('"').strip("'"))

    if ckpt_path.is_file():
        info = f"最後更新 {datetime.fromtimestamp(ckpt_path.stat().st_mtime):%Y-%m-%d %H:%M}"
        if not (ckpt_path.parent / "config.json").exists():
            info += "；同資料夾沒有 config.json，色彩校正會用預設的 meanstd"
        st.caption(info)
    else:
        st.warning("找不到這個權重檔，請確認路徑。")

    n_future = st.number_input("預測年數", min_value=1, max_value=12, value=6, key="n_future")

    run_forecast = st.button(
        "開始預測", type="primary", key="run_forecast",
        disabled=len(valid_files) == 0 or not ckpt_path.is_file(),
    )

    if run_forecast:
        with tempfile.TemporaryDirectory() as tmp:
            for f in valid_files:
                (Path(tmp) / f.name).write_bytes(f.getbuffer())

            shutil.rmtree(OUT_DIR, ignore_errors=True)

            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            with st.spinner("模型推論中，請稍候..."):
                result = subprocess.run(
                    [
                        sys.executable, str(INFER_SCRIPT),
                        "--data_dir", tmp,
                        "--ckpt", str(ckpt_path),
                        "--out_dir", str(OUT_DIR),
                        "--n_future", str(int(n_future)),
                    ],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    cwd=str(INFER_SCRIPT.parent), env=env,
                )

        if result.returncode == 0:
            st.success("預測完成")
            for p in sorted(OUT_DIR.glob("pred_*.png")):
                st.image(str(p), caption=p.stem.replace("pred_", "預測 "), width="stretch")
            for p in sorted(OUT_DIR.glob("std_*.png")):
                st.image(str(p), caption=p.stem.replace("std_", "標準差 "), width="stretch")
            with st.expander("執行紀錄"):
                st.text(result.stdout)
        else:
            st.error("程式執行失敗")
            st.text_area("錯誤訊息", result.stderr or result.stdout, height=300)

    st.markdown(
        """
        **注意事項**：
        1. 請確保上傳的影像檔名為年份，例如 `2013.png`、`2024.jpg`，以便系統正確辨識年份。
        2. 預測年數請設定為 1 到 12 年之間(最佳預測年數為**6**年)，系統將以已訓練的生成模型預測未來影像。
        3. 預測結果將包含每一年份的預測影像以及對應的標準差圖，標準差圖可用於評估模型對該年份預測的不確定性。
        4. 若程式執行失敗，請檢查上傳的影像格式及檔名是否正確，並確認權重檔案是否存在。
        """)

# ============================================================
# 新增衛星影像 分頁 — 新增到「現有資料集」, 並可選擇重新訓練模型
# 貼在 tab2 (with main_tab2:) 區塊之後。
#
# 資料集結構 (與 resnet_dcgan_cijin.py 相同):
#   cijin/海水/2013.jpg ... 2026.jpg
#   cijin/風車/2013.jpg ... 2026.jpg
# 模型檔位置:  resnet_DCGAN/generator.pt + config.json  (岸線變遷分析分頁使用的那一份)
# 訓練腳本:    resnet_DCGAN/resnet_dcgan_cijin.py
# ============================================================


_APP_DIR = Path(__file__).resolve().parent

# >>> 改成你的現有資料集資料夾 (也可以在分頁裡直接改) <<<
DEFAULT_DATASET_DIR = _APP_DIR / "cijin"

MODEL_DIR = _APP_DIR / "train_runs"                     # 目前使用中的 generator.pt / config.json
TRAIN_SCRIPT = _APP_DIR / "resnet_dcgan_cijin.py"
TRAIN_RUNS_DIR = MODEL_DIR / "train_runs"                   # 每次訓練各自一個資料夾, 不會動到使用中的模型

SAT_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


# ------------------------- 資料集輔助 -------------------------
def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower().lstrip(".") in SAT_EXTS


def _year_files(folder) -> dict:
    """回傳 {年份: 路徑}, 檔名不是純數字的檔案略過 (與訓練腳本的讀法一致)。"""
    out = {}
    for p in Path(folder).iterdir():
        if _is_img(p) and p.stem.isdigit():
            out[int(p.stem)] = p
    return dict(sorted(out.items()))


def _discover_sites(root) -> dict:
    """子資料夾 (內含年份圖片) = 地點; 根目錄本身有年份圖片則視為單一地點。"""
    root, sites = Path(root), {}
    if not root.is_dir():
        return sites
    if _year_files(root):
        sites[root.name or "site"] = root
    for d in sorted(root.iterdir()):
        if d.is_dir() and _year_files(d):
            sites[d.name] = d
    return sites


def _family(ext: str) -> str:
    ext = ext.lower()
    return {".jpeg": ".jpg", ".tiff": ".tif"}.get(ext, ext)


def _dominant_ext(files: dict) -> str:
    c = Counter(_family(p.suffix) for p in files.values())
    return c.most_common(1)[0][0] if c else ".png"


def _save_into(folder: Path, ext: str, uploaded, img, year: int):
    """依該地點既有的副檔名存檔 (副檔名相同就原檔寫入, 不重新壓縮); 同年份不會留下兩個檔。"""
    dest = folder / f"{year}{ext}"
    if _family(Path(uploaded.name).suffix) == ext:
        dest.write_bytes(uploaded.getbuffer())
    elif ext == ".jpg":
        img.save(dest, quality=95)
    else:
        img.save(dest)
    for old in folder.iterdir():                     # 同年份、不同副檔名的舊檔 -> 刪除
        if _is_img(old) and old.stem == str(year) and old != dest:
            old.unlink()


# ------------------------- 訓練輔助 -------------------------
def _apply_model(run_dir: Path) -> Path:
    """把這次訓練的模型套用為「使用中的模型」; 舊的先備份到 resnet_DCGAN/backup/<時間>/。"""
    bak = MODEL_DIR / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    bak.mkdir(parents=True, exist_ok=True)
    for name in ("generator.pt", "config.json"):
        old = MODEL_DIR / name
        if old.exists():
            shutil.copy2(old, bak / name)
        shutil.copy2(run_dir / name, MODEL_DIR / name)
    return bak


def _start_training(cmd: list, run_dir: Path, mode: str, epochs: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(TRAIN_SCRIPT.parent), env=env,
        )
    st.session_state.train_proc = proc
    st.session_state.train_log = log_path
    st.session_state.train_out = run_dir
    st.session_state.train_mode = mode
    st.session_state.train_epochs = epochs
    st.session_state.train_t0 = time.time()
    st.session_state.train_was_running = True


def _train_status():
    """顯示訓練進度 / 結果。訓練中以 fragment 每 3 秒自動更新, 不會因為重新整理而中斷訓練。"""
    proc = st.session_state.get("train_proc")
    if proc is None:
        return
    log_path = st.session_state.train_log
    out_dir = st.session_state.train_out
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    found = re.findall(r"\[epoch\s+(\d+)/(\d+)\]", text)
    running = proc.poll() is None

    if not running and st.session_state.get("train_was_running"):
        st.session_state.train_was_running = False
        st.rerun()                                   # 結束 -> 重畫整頁, 停止自動更新

    if running:
        done, total = (int(found[-1][0]), int(found[-1][1])) if found else (0, st.session_state.train_epochs)
        elapsed = int(time.time() - st.session_state.train_t0)
        st.progress(
            min(done / max(total, 1), 1.0),
            text=f"訓練中… epoch {done}/{total}（已經過 {elapsed // 60} 分 {elapsed % 60} 秒）"
            if found else "載入資料與初始化中…",
        )
        st.warning("若有配置GPU，請確定 [info] device = cuda，否則請重新執行以下程式。"
                    "pip uninstall torch torchvision torchaudio -y\n"
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/(依GPU版本安裝對應的版本)")

        if st.button("停止訓練", key="train_stop"):
            proc.terminate()
            st.rerun()
    elif proc.returncode == 0 and (out_dir / "generator.pt").exists():
        st.success(f"訓練完成，模型已存到：{out_dir}")
        if st.session_state.train_mode == "formal":
            if st.button("套用為目前使用的模型", type="primary", key="train_apply"):
                bak = _apply_model(out_dir)
                st.success(
                    "已套用。「岸線變遷分析」分頁之後會使用新模型；"
                    f"舊模型已備份到 {bak}"
                )
        else:
            st.info(
                "這是回測模型（沒有使用最後幾年的影像訓練），只用來驗證，不能當正式模型。"
                "請看下方紀錄中的 MAE 與「模型 vs 持續性基準」比較；"
                f"預測與對照圖在 {out_dir}。"
            )
    else:
        st.error("訓練中止或失敗，請看下方紀錄。")

    st.code("\n".join(text.splitlines()[-25:]) or "（尚無輸出）", language="text")


if "sat_uploader_n" not in st.session_state:
    st.session_state.sat_uploader_n = 0              # 儲存後用來重設上傳元件

with main_tab3:
    st.header("新增衛星影像及重新訓練模型")
    st.markdown(
        "把新的年份影像直接加進**現有資料集**。同一地點的影像尺寸必須與現有影像一致，"
        "檔案會以「年份.副檔名」存放，與訓練資料同格式。新增後可在最下方重新訓練模型。"
    )
    st.warning(
        "這會直接寫入訓練用的資料夾。覆蓋舊影像前建議先備份資料集；"
        "新增影像不會自動更新已訓練好的模型，需要重新訓練才會用到。"
    )

    if st.session_state.get("sat_flash"):
        st.success(st.session_state.pop("sat_flash"))

    # ---------------- 資料集與地點 ----------------
    dataset_dir = st.text_input("資料集資料夾", value=str(DEFAULT_DATASET_DIR), key="sat_dataset_dir")
    sites = _discover_sites(dataset_dir)

    if not sites:
        st.error(f"在「{dataset_dir}」找不到任何「年份.jpg」格式的影像，請確認資料集路徑。")
    else:
        site = st.selectbox("地點", options=list(sites), key="sat_site_choice")
        folder = sites[site]
        existing = _year_files(folder)
        years_have = list(existing)
        target_ext = _dominant_ext(existing)
        with Image.open(existing[years_have[0]]) as _im:
            ref_size = _im.size

        st.caption(
            f"「{site}」目前有 {len(years_have)} 張（{years_have[0]}–{years_have[-1]}），"
            f"尺寸 {ref_size[0]}×{ref_size[1]}，格式 {target_ext}。"
        )
        gaps_now = [y for y in range(years_have[0], years_have[-1] + 1) if y not in existing]
        if gaps_now:
            st.caption(f"目前缺少年份：{gaps_now}")

        # ---------------- 上傳 ----------------
        files = st.file_uploader(
            "選擇要新增的衛星影像（可多選）",
            type=list(SAT_EXTS),
            accept_multiple_files=True,
            key=f"sat_uploader_{st.session_state.sat_uploader_n}",
        )

        items = []                                   # [(檔案, PIL 影像, 年份)]
        if files:
            st.markdown("**確認每張影像的年份**（自動從檔名抓 4 位數年份，抓不到請手動填）")
            for i, f in enumerate(files):
                m = re.search(r"(19|20)\d{2}", Path(f.name).stem)
                default_year = int(m.group(0)) if m else datetime.now().year
                img = Image.open(f).convert("RGB")
                c_img, c_year, c_info = st.columns([1, 1, 3])
                with c_img:
                    st.image(img, width=120)
                with c_year:
                    year = st.number_input(
                        "年份", min_value=1900, max_value=2100, value=default_year,
                        key=f"sat_year_{st.session_state.sat_uploader_n}_{i}",
                        label_visibility="collapsed",
                    )
                with c_info:
                    st.caption(f"{f.name} ・ {img.size[0]}×{img.size[1]}")
                items.append((f, img, int(year)))

            # ---------------- 檢查 ----------------
            problems = []
            years = [y for _, _, y in items]

            dup = sorted({y for y in years if years.count(y) > 1})
            if dup:
                problems.append(f"本次上傳有重複的年份：{dup}")

            bad_size = [f.name for f, img, _ in items if img.size != ref_size]
            if bad_size:
                problems.append(
                    f"以下影像尺寸與資料集（{ref_size[0]}×{ref_size[1]}）不符：{bad_size}"
                )

            clash = sorted(y for y in set(years) if y in existing)
            overwrite = False
            if clash:
                overwrite = st.checkbox(
                    f"年份 {clash} 在資料集中已存在，確定要覆蓋", key="sat_overwrite"
                )

            for msg in problems:
                st.error(msg)

            all_years = sorted(set(years_have) | set(years))
            gaps_after = [y for y in range(all_years[0], all_years[-1] + 1) if y not in all_years]
            if gaps_after and not problems:
                st.info(f"加入後資料集仍缺少年份：{gaps_after}（訓練時跨越缺年的序列不會被使用）")

            can_save = (not problems) and (not clash or overwrite)
            if st.button("新增到資料集", type="primary", key="sat_save", disabled=not can_save):
                for f, img, y in items:
                    _save_into(folder, target_ext, f, img, y)
                st.session_state.sat_flash = (
                    f"已將 {len(items)} 張影像（{sorted(years)}）新增到「{site}」，"
                    f"共 {len(set(years_have) | set(years))} 張。"
                    "如要讓模型使用新影像，請到最下方「重新訓練模型」。"
                )
                st.session_state.sat_uploader_n += 1
                st.rerun()

        # ---------------- 目前資料集內容 ----------------
        st.markdown("---")
        st.subheader("資料集內容")
        for s, fol in sites.items():
            yf = _year_files(fol)
            ys = list(yf)
            with st.expander(f"{s}（{len(ys)} 張，{ys[0]}–{ys[-1]}）", expanded=False):
                sel = st.selectbox("預覽年份", ys, key=f"sat_prev_{s}")
                st.image(str(yf[sel]), caption=f"{s} {sel}", width="stretch")

        # ---------------- 重新訓練模型 ----------------
        st.markdown("---")
        st.subheader("重新訓練ResNet-DCGAN模型（選用）")

        cur_pt = MODEL_DIR / "generator.pt"
        if cur_pt.exists():
            st.caption(
                "目前使用的模型：generator.pt，最後更新 "
                + datetime.fromtimestamp(cur_pt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            )
        else:
            st.caption("目前還沒有 generator.pt。")

        proc_now = st.session_state.get("train_proc")
        is_running = proc_now is not None and proc_now.poll() is None

        if not torch.cuda.is_available():
            st.warning(
                "偵測不到 GPU（CUDA），會用 CPU 訓練，速度非常慢（預設參數可能要數小時以上）。"
                "建議先把 epoch 調小做測試，或改在有 GPU 的機器上訓練。"
            )
        if not TRAIN_SCRIPT.exists():
            st.error(f"找不到訓練腳本：{TRAIN_SCRIPT}")

        train_mode_label = st.radio(
            "訓練方式",
            options=["正式訓練（使用全部年份）", "回測驗證（留出最後幾年，用來檢驗模型準不準）"],
            key="train_mode_label",
            disabled=is_running,
        )
        is_formal = train_mode_label.startswith("正式")

        holdout = 0
        if not is_formal:
            holdout = st.number_input(
                "留出最後幾年做驗證", min_value=1, max_value=5, value=2, key="train_holdout",
                disabled=is_running,
            )

        chosen_sites = st.multiselect(
            "使用的地點", options=list(sites), default=list(sites), key="train_sites",
            disabled=is_running,
        )

        with st.expander("進階參數", expanded=False):
            a1, a2, a3 = st.columns(3)
            with a1:
                epochs = st.number_input("epochs", 1, 1000, 150, key="train_epochs_in", disabled=is_running)
                k = st.number_input("輸入年數 k", 1, 6, 3, key="train_k", disabled=is_running)
            with a2:
                samples_per_epoch = st.number_input(
                    "每個 epoch 取樣數", 50, 5000, 800, step=50, key="train_spe", disabled=is_running
                )
                batch_size = st.number_input("batch size", 1, 64, 8, key="train_bs", disabled=is_running)
            with a3:
                patch_size = st.selectbox(
                    "訓練圖塊大小", [128, 192, 256, 384], index=2, key="train_patch", disabled=is_running
                )
                color_match = st.selectbox(
                    "色彩校正", ["meanstd", "none"], key="train_cm", disabled=is_running
                )
            use_gan = st.checkbox("使用 GAN（取消則為純 L1）", value=True, key="train_gan", disabled=is_running)
            flip = st.checkbox("左右翻轉增強", value=False, key="train_flip", disabled=is_running)

        if st.button(
            "開始訓練", type="primary", key="train_start",
            disabled=is_running or not TRAIN_SCRIPT.exists() or not chosen_sites,
        ):
            run_dir = TRAIN_RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            cmd = [
                sys.executable, "-u", str(TRAIN_SCRIPT),
                "--data_dir", str(dataset_dir),
                "--out_dir", str(run_dir),
                "--k", str(int(k)),
                "--epochs", str(int(epochs)),
                "--samples_per_epoch", str(int(samples_per_epoch)),
                "--batch_size", str(int(batch_size)),
                "--patch_w", str(int(patch_size)),
                "--patch_h", str(int(patch_size)),
                #"--color_match", color_match,
                "--log_every", str(max(1, int(epochs) // 40)),
            ]
            if set(chosen_sites) != set(sites):
                cmd += ["--sites", *chosen_sites]
            if not use_gan:
                cmd += ["--adv_weight", "0"]
            if flip:
                cmd.append("--flip")
            if is_formal:
                # 訓練腳本訓練完會順便預測一次, 這裡只預測 1 年 1 條軌跡, 把額外時間降到最低
                cmd += ["--holdout", "0", "--n_future", "1", "--n_samples", "1", "--infer_batch", "1"]
            else:
                cmd += ["--holdout", str(int(holdout)), "--n_samples", "10", "--infer_batch", "2"]

            _start_training(cmd, run_dir, "formal" if is_formal else "backtest", int(epochs))
            st.rerun()

        # 訓練中用 fragment 自動更新; 沒在訓練就只畫一次
        if is_running:
            st.fragment(run_every=3)(_train_status)()
        else:
            _train_status()

with main_tab4:
    st.header("關於本系統")
    st.markdown(
        """
        本系統為 台灣世曦 **港灣部** *深度學習於岸線變遷之應用與分析預測模組之研發－以旗津海岸為例 (IV~VI)*
        一案之應用平台，旨在提供 DeepLabV3-ResNet101 模型的進行海岸偵測功能，並結合DiffusionEdge模型進行岸線變遷分析。

        **岸線偵測模組** 採用 DeepLabV3-ResNet101 進行語意分割，並支援 AI 輔助人工修正（操作方式類似 [CVAT](https://www.cvat.ai/) 的語意分割標註工具）。

        **岸線變遷分析模組** 採用 Resnet-DCGAN 生成模型，使用多年份影像作為輸入，預測未來年份的影像。

        本系統由 [KSE108](https://github.com/ksuee108/Cijin) 開發。

        """)