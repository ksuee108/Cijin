
import cv2
import glob
import os
import numpy as np

img_list = glob.glob(
    r"origin_image\infer_future\pred_*.png"
)

# 設定輸出資料夾
output_dir = r"origin_image\infer_future\binary"
os.makedirs(output_dir, exist_ok=True)

for img_path in img_list:

    # 1. 讀取彩色圖片
    img = cv2.imread(img_path)

    if img is None:
        print(f"無法讀取圖片：{img_path}")
        continue

    blur = cv2.GaussianBlur(img, (5, 5), 0)   # 模糊縮小，避免吃掉細結構

    # ---------- 1. 陸地／水（和之前一樣，但不再做開運算與面積過濾）----------
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    _, water = cv2.threshold(lab[:, :, 2], 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    land = cv2.bitwise_not(water)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, k)   # 只補洞

    # ---------- 2. 單獨抽出設施：在水域範圍內找「亮的」東西 ----------
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] > 130) & (hsv[:, :, 1] < 80)   # 亮且低飽和 = 灰白水泥
    in_water = cv2.dilate(water, np.ones((3, 3), np.uint8)) > 0

    # 只取「被水包圍」的亮物體，避免把陸地上的建物誤判
    facility = (bright & in_water).astype(np.uint8) * 255
    facility = cv2.morphologyEx(facility, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 去掉極小雜點（門檻設低，避免誤刪小設施）
    n, lab_, st, _ = cv2.connectedComponentsWithStats(facility)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 30:
            facility[lab_ == i] = 0

    # ---------- 3. 合併 ----------
    result = cv2.bitwise_or(land, facility)
    output_path = os.path.join(
        output_dir,
        f"binary_{os.path.basename(img_path)}"
    )

    cv2.imwrite(output_path, result)

    print(f"已儲存：{output_path}")
        # 半透明紅色填滿陸地 + 黃色輪廓
    overlay = img.copy()
    overlay[result > 0] = (0, 0, 255)                  # BGR：紅
    vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, f"st_{os.path.basename(img_path)}"), vis)

print("所有圖片處理完成！")
