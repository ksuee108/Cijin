from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_reference_images(dataset_dir, year=2024):
    """
    在資料集資料夾底下找「<year>.<副檔名>」的影像。
    回傳 {顯示名稱: 路徑}，例如 {"海水/2024.jpg": Path(...), "風車/2024.jpg": Path(...)}。
    根目錄本身若有 2024.xxx 也會列入。
    """

    root = Path(dataset_dir)
    found = {}

    if not root.is_dir():
        return found

    folders = [root] + sorted(d for d in root.iterdir() if d.is_dir())

    for folder in folders:
        for ext in IMG_EXTS:
            p = folder / f"{year}{ext}"
            if p.is_file():
                label = p.name if folder == root else f"{folder.name}/{p.name}"
                found[label] = p
                break

    return found


def _match_channel(source, reference):
    """單一通道的直方圖匹配（CDF 對 CDF 的對應）。"""

    src_values, src_idx, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / reference.size

    mapped = np.interp(src_cdf, ref_cdf, ref_values)

    return mapped[src_idx].reshape(source.shape)


def match_histogram_to_reference(image, reference):
    """
    把 image（PIL 或 RGB ndarray）的各通道直方圖對齊到 reference。
    回傳 uint8 的 RGB ndarray，尺寸與 image 相同
    （reference 的尺寸可以不同，只用它的像素分佈）。
    """

    src = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
    ref = np.array(reference.convert("RGB")) if isinstance(reference, Image.Image) else np.asarray(reference)

    out = np.empty_like(src, dtype=np.uint8)

    for c in range(3):
        matched = _match_channel(src[:, :, c], ref[:, :, c])
        out[:, :, c] = np.clip(np.rint(matched), 0, 255).astype(np.uint8)

    return out
