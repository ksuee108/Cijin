import cv2
import numpy as np
import streamlit as st
import torch
import albumentations as A
from PIL import Image
from config import LABEL_COLORS_LIST
import torch
import torch.nn as nn
import torch.nn.functional as F


# 假設 class index 0 為「背景 / 其他」，可作為橡皮擦使用
BACKGROUND_CLASS_INDEX = 0

# ============================================================
# DeepLabV3-ResNet101 模型
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision.models.segmentation import deeplabv3_resnet101
 
def prepare_101_model(num_classes=2):
    print("Preparing DeepLabV3 ResNet101 model...")
    model = deeplabv3_resnet101(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(256, num_classes, kernel_size=1)
)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

@st.cache_resource
def load_model():
    model_path = r"model\best_model-DeepLabV3-ResNet101.pth"
    model = prepare_101_model()

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model





# ============================================================
# 影像處理函式
# ============================================================

def adjust_image(image, brightness=0, contrast=0, hue=0, saturation=0, value=0):

    img = np.array(image.convert("RGB"))

    alpha = 1.0 + contrast / 100.0
    beta = brightness

    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

    hsv[:, :, 0] += int(hue // 2)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)

    hsv[:, :, 1] += saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    hsv[:, :, 2] += value
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    hsv = hsv.astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return Image.fromarray(img)


# ============================================================
# DeepLabV3 preprocessing
# ============================================================

def preprocess_image(image):

    image_np = np.array(image.convert("RGB"))

    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.45734706, 0.43338275, 0.40058118],
                std=[0.23965294, 0.23532275, 0.2398498],
                max_pixel_value=255.0
            )
        ]
    )

    transformed = transform(image=image_np)

    image_np = transformed["image"]
    image_np = np.transpose(image_np, (2, 0, 1))

    image_tensor = torch.from_numpy(image_np).float()

    return image_tensor


# ============================================================
# DeepLabV3 ROI inference
# ============================================================

@torch.no_grad()
def predict_roi(roi_image, model, patch_size=512):

    roi_np = np.array(roi_image.convert("RGB"))
    roi_h, roi_w = roi_np.shape[:2]

    prediction = np.zeros((roi_h, roi_w), dtype=np.uint8)

    # 補到 patch_size 的整數倍，確保每一塊 patch 進模型前都是完整的
    # patch_size x patch_size，避免最後一排/列被截斷成非正方形尺寸。
    pad_h = (patch_size - roi_h % patch_size) % patch_size
    pad_w = (patch_size - roi_w % patch_size) % patch_size

    padded = cv2.copyMakeBorder(
        roi_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )

    padded_h, padded_w = padded.shape[:2]

    for y in range(0, padded_h, patch_size):
        for x in range(0, padded_w, patch_size):

            patch = padded[y:y + patch_size, x:x + patch_size]
            patch_pil = Image.fromarray(patch)

            tensor = preprocess_image(patch_pil)
            tensor = tensor.unsqueeze(0).to(device)

            output = model(tensor)

            if isinstance(output, dict):
                output = output["out"]

            pred = torch.argmax(output, dim=1)[0]
            pred = pred.cpu().numpy().astype(np.uint8)

            # 模型輸出固定為 patch_size x patch_size，若與這塊 patch 實際
            # 尺寸不同（理論上補完後不會發生，這裡作為保險），先 resize 對齊。
            patch_h, patch_w = patch.shape[:2]

            if pred.shape != (patch_h, patch_w):
                pred = cv2.resize(
                    pred,
                    (patch_w, patch_h),
                    interpolation=cv2.INTER_NEAREST
                )

            # 寫回 prediction 時要用「未補邊」的 roi_h / roi_w 當邊界，
            # 不能用補邊後的 padded_h / padded_w，否則最後一塊 patch
            # 會因為 prediction 陣列本身較小而被自動截斷，造成左右兩邊
            # 形狀對不上而 broadcast 失敗。
            h_end = min(y + patch_size, roi_h)
            w_end = min(x + patch_size, roi_w)

            if h_end <= y or w_end <= x:
                # 這塊 patch 整塊都落在補邊區域內，沒有對應的原始像素可寫回
                continue

            prediction[y:h_end, x:w_end] = pred[:h_end - y, :w_end - x]

    prediction = prediction[:roi_h, :roi_w]

    return prediction


# ============================================================
# Decode segmentation mask
# ============================================================

def decode_segmap(mask, label_colors_list):

    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for label_index, color in enumerate(label_colors_list):
        rgb[mask == label_index] = color

    return rgb


# ============================================================
# Overlay（通用版本：疊所有前景類別，不限 ROI 範圍）
# ============================================================

def create_overlay(original_image, prediction, roi=None, alpha=0.5):

    original = np.array(original_image.convert("RGB"))
    segmentation_rgb = decode_segmap(prediction, LABEL_COLORS_LIST)

    overlay = original.copy()

    foreground_mask = prediction != BACKGROUND_CLASS_INDEX

    overlay[foreground_mask] = (
        original[foreground_mask] * (1 - alpha)
        + segmentation_rgb[foreground_mask] * alpha
    ).astype(np.uint8)

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 3)

    return overlay


# ============================================================
# 人工修正：從 fabric.js path 物件取出座標點
# ============================================================

def extract_path_points(obj):
    """
    從 streamlit-drawable-canvas 回傳的 fabric.js 物件中取出座標點列表。
    適用於 freedraw（筆刷）與 polygon（多邊形）兩種 drawing_mode。
    """

    path_data = obj.get("path", [])
    points = []

    for command in path_data:

        if len(command) < 2:
            continue

        nums = [c for c in command[1:] if isinstance(c, (int, float))]

        if len(nums) >= 2:
            x, y = nums[-2], nums[-1]
            points.append((x, y))

    return points


def rasterize_polygon(mask, points, class_index):

    if len(points) < 3:
        return

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], int(class_index))


def rasterize_brush(mask, points, class_index, thickness):

    if len(points) == 0:
        return

    thickness = max(1, int(round(thickness)))

    if len(points) == 1:
        x, y = points[0]
        cv2.circle(mask, (int(x), int(y)), thickness // 2, int(class_index), -1)
        return

    pts = np.array(points, dtype=np.int32)

    cv2.polylines(
        mask,
        [pts],
        isClosed=False,
        color=int(class_index),
        thickness=thickness,
        lineType=cv2.LINE_8
    )

    # 補上端點圓形，避免線段轉角處出現空隙
    for x, y in points:
        cv2.circle(mask, (int(x), int(y)), thickness // 2, int(class_index), -1)


def apply_correction_objects(mask, objects, scale_x, scale_y, tool_mode, class_index, brush_size):

    working_mask = mask.copy()
    brush_thickness_original = max(1, int(round(brush_size * (scale_x + scale_y) / 2)))

    for obj in objects:

        points = extract_path_points(obj)

        if len(points) == 0:
            continue

        scaled_points = [(px * scale_x, py * scale_y) for px, py in points]

        if tool_mode == "polygon":
            rasterize_polygon(working_mask, scaled_points, class_index)
        else:
            rasterize_brush(
                working_mask,
                scaled_points,
                class_index,
                brush_thickness_original
            )

    return working_mask


def class_legend_markdown(all_classes, label_colors_list):

    rows = []

    for idx, name in enumerate(all_classes):

        if idx >= len(label_colors_list):
            break

        color = label_colors_list[idx]
        color_css = f"rgb({color[0]},{color[1]},{color[2]})"

        rows.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background-color:{color_css};margin-right:6px;border:1px solid #999;"></span>'
            f'{idx}: {name}</span>'
        )

    return "<div style='margin-bottom:8px;'>" + "".join(rows) + "</div>"


# ============================================================
# 從畫布結果解析出最後一個矩形框（畫布座標 → 原始影像座標）
# ============================================================

def parse_rect_bbox_from_canvas(canvas_result, img_width, img_height, disp_width, disp_height):

    if canvas_result.json_data is None:
        return None

    objects = canvas_result.json_data.get("objects", [])
    rectangles = [obj for obj in objects if obj.get("type", "").lower() == "rect"]

    if len(rectangles) == 0:
        return None

    rect = rectangles[-1]

    left = float(rect.get("left", 0))
    top = float(rect.get("top", 0))
    rect_width = float(rect.get("width", 0))
    rect_height = float(rect.get("height", 0))

    scale_x_fabric = float(rect.get("scaleX", 1))
    scale_y_fabric = float(rect.get("scaleY", 1))

    rect_width *= scale_x_fabric
    rect_height *= scale_y_fabric

    sx = img_width / float(disp_width)
    sy = img_height / float(disp_height)

    x1 = int(round(left * sx))
    y1 = int(round(top * sy))
    x2 = int(round((left + rect_width) * sx))
    y2 = int(round((top + rect_height) * sy))

    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)