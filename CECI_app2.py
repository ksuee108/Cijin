import io
import numpy as np
import streamlit as st
import torch
import albumentations as A

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from config import ALL_CLASSES, LABEL_COLORS_LIST

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
# Sidebar
# ============================================================

#st.sidebar.title("AI 模型")

#st.sidebar.info(
#    "目前系統固定使用：\n\n"
#    "**DeepLabV3-ResNet101**"
#)

# ============================================================
# Title
# ============================================================

st.title("台灣世曦 港灣部 岸線偵測系統")

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
            roi_np = image_np[y1:y2, x1:x2]
            roi_image = Image.fromarray(roi_np)

            progress_text = st.empty()
            progress_text.info("DeepLabV3-ResNet101 正在進行 ROI 語意分割...")

            prediction_roi = predict_roi(roi_image, model, patch_size=512)

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