import streamlit as st
import pandas as pd
import os
import torch
import os
import albumentations as A
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from model_2doutput import (
    prepare_50_model, 
    prepare_101_model, 
    prepare_mobilenet_model, 
    UNet,
    CNN,
    ConvLSTM
)
from config import ALL_CLASSES, LABEL_COLORS_LIST

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

user_home = os.path.expanduser("~")

st.set_page_config(page_title="台灣世曦 港灣部 岸線偵測系統", layout="wide")
st.title("台灣世曦 港灣部 岸線偵測系統")

st.markdown("""本系統為 台灣世曦 **港灣部** *深度學習於岸線變遷之應用與分析預測模組之研發－以旗津海岸為例 (V)*
            一案之應用平台，旨在提供CNN，U-NET，Conv-LSTM，DeepLabV3-ResNet50，DeepLabV3-ResNet101，DeepLabV3-outputs_MobileNet六種模型的岸線偵測功能。""")

st.sidebar.title("選擇模型與上傳影像")
model_option = st.sidebar.multiselect(
    "請選擇欲使用之模型:", ["CNN", "U-NET", "Conv-LSTM", "DeepLabV3-ResNet50", "DeepLabV3-ResNet101", "DeepLabV3-MobileNet"])

_defaults = {'brightness': 0, 'contrast': 0, 'hue': 0, 'saturation': 0, 'value': 0}

def _reset_sliders():
    for _k, _v in _defaults.items():
        st.session_state[_k] = _v

brightness = st.sidebar.slider("亮度增減 (%)", -50, 50, st.session_state.get('brightness', 0), key='brightness')       # 對應 -0.5 ~ +0.5
contrast   = st.sidebar.slider("對比度增減 (%)", -50, 50, st.session_state.get('contrast', 0), key='contrast')     # 對應 -0.5 ~ +0.5
hue        = st.sidebar.slider("色相偏移", -180, 180, st.session_state.get('hue', 0), key='hue')
saturation = st.sidebar.slider("飽和度偏移", -100, 100, st.session_state.get('saturation', 0), key='saturation')
value      = st.sidebar.slider("明度偏移", -100, 100, st.session_state.get('value', 0), key='value')


st.sidebar.button("重設參數", on_click=_reset_sliders)

uploaded_files = st.file_uploader(
    "請上傳欲偵測之影像:", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

def decode_segmap(mask, label_colors_list):
    """
    將 segmentation mask 的每個 pixel 類別 index 映射為 RGB 顏色。
    """
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in enumerate(label_colors_list):
        rgb[mask == class_idx] = color
    return rgb

model_dict = {
    'U-NET': lambda num_classes: UNet(input_channels=3, output_channels=2),
    'CNN': lambda num_classes: CNN(),
    'Conv-LSTM': lambda num_classes: ConvLSTM(input_dim=3, hidden_dims=[16, 32, 63, 32, 2]), #CNN-LSTM = ConvLSTM
    'DeepLabV3-ResNet50': prepare_50_model, #DeepLabV3-ResNet50
    'DeepLabV3-ResNet101': prepare_101_model, #DeepLabV3-ResNet101
    'DeepLabV3-MobileNet': prepare_mobilenet_model, #DeepLabV3-MobileNet
}

all_results = []  # 用來存所有模型的結果

if uploaded_files and model_option:
    st.sidebar.success(f"已選擇 {len(uploaded_files)} 張影像，使用模型: {', '.join(model_option)}")
    
    for uploaded_file in uploaded_files:
        st.subheader(f"影像: {uploaded_file.name}")
        aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(brightness/100, brightness/100),
                contrast_limit=(contrast/100, contrast/100),
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=hue,
                sat_shift_limit=saturation,
                val_shift_limit=value,
                p=1.0
            )
    ])
        original_image = Image.open(uploaded_file).convert("RGB")
        augmented_np = aug(image=np.array(original_image))["image"]
        original_image = Image.fromarray(augmented_np)
        st.image(original_image, caption=f"上傳的影像: {uploaded_file.name}", use_column_width=True)
        
        if st.button("開始偵測"):
            st.markdown("### 偵測結果:")
            for model in model_option:
                out_dir_model = os.path.join("model", 'best_model-{}.pth'.format(model))

                net  = model_dict[model](num_classes=len(ALL_CLASSES))
                checkpoint = torch.load(out_dir_model, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    net.load_state_dict(checkpoint["model_state_dict"])
                else:
                    net.load_state_dict(checkpoint)
                net = net.to(device)
                net.eval()
                aug_pipeline = A.Compose([
                    A.Resize(128, 128),                     # 調整影像大小
                ])
                metrics_list = []
                patch_size = 128

                h, w = Image.open(uploaded_file).size
                original_np = np.array(original_image)

                pad_h = (patch_size - h % patch_size) % patch_size
                pad_w = (patch_size - w % patch_size) % patch_size
                padded = np.pad(original_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                ph, pw = padded.shape[:2]

                # 用來儲存所有小 patch 預測結果
                pred_full = np.zeros((ph, pw), dtype=np.uint8)

                # 遍歷每個 256x256 區塊
                for y in range(0, ph, patch_size):
                    for x in range(0, pw, patch_size):
                        patch = padded[y:y+patch_size, x:x+patch_size]
                        augmented = aug_pipeline(image=patch)
                        patch_aug = augmented["image"]

                        transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(
                                mean=[0.45734706, 0.43338275, 0.40058118],
                                std=[0.23965294, 0.23532275, 0.2398498]
                            )
                        ])
                        input_tensor = transform(patch_aug).unsqueeze(0).to(device)
                    
                        with torch.no_grad():
                            if isinstance(net, ConvLSTM):
                                input_tensor = input_tensor.unsqueeze(1)  # [B, 1, C, H, W]
                                output = net(input_tensor)
                            else:
                                output = net(input_tensor)
                                if isinstance(output, dict):  # DeepLabV3
                                    output = output['out']

                            pred_patch = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                        y_end = min(y + patch_size, ph)
                        x_end = min(x + patch_size, pw)

                        pred_patch = pred_patch[:(y_end - y), :(x_end - x)]

                        pred_full[y:y_end, x:x_end] = pred_patch

                # 移除 padding 回到原圖大小
                pred = pred_full[:h, :w]

                if model not in ['DeepLabV3-ResNet50', 'DeepLabV3-ResNet101', 'DeepLabV3-MobileNet']:
                        pred = 1 - pred

                color_seg = decode_segmap(pred, LABEL_COLORS_LIST)

                # Resize 分割圖回原圖大小
                color_seg_resized = Image.fromarray(color_seg).resize(original_image.size, resample=Image.NEAREST)
                

                # 儲存分割彩圖
                save_path = f"seg_output_{model}.png"
                color_seg_resized.save(save_path)
                
                # 疊加原圖與彩色分割圖
                original_np = np.array(original_image.convert("RGB")).astype(np.uint8)
                pred_resized = cv2.resize(
                    pred,
                    (original_np.shape[1], original_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                #augmented = aug(image=original_np)["image"]

                class0_mask = (pred_resized == 1)

                overlay = np.zeros_like(original_np, dtype=np.uint8)
                
                overlay[class0_mask] = LABEL_COLORS_LIST[1]
                overlay[class0_mask] = (255, 128, 0)

                alpha = 0.5
                blended = original_np.copy()
                blended[class0_mask] = (
                    original_np[class0_mask] * (1 - alpha) +
                    overlay[class0_mask] * alpha
                ).astype(np.uint8)

                blended_output_path =  f"seg_overlay_{model}.png"
                cv2.imwrite(blended_output_path, blended)
                

                st.markdown(f"#### 使用模型: {model}")
                st.image(blended_output_path, caption=f"{model} 偵測結果 for {uploaded_file.name}", use_column_width=True)
