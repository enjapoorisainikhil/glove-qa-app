"""
visual_diff.py — aligns two images and finds differences
"""
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class DiffResult:
    heatmap_bgr: np.ndarray
    overlay_bgr: np.ndarray
    aligned_supplier: np.ndarray
    diff_score: float
    diff_pixel_count: int
    total_pixels: int

def _resize_match(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    return cv2.resize(tgt, (w, h), interpolation=cv2.INTER_LANCZOS4)

def align_images(master: np.ndarray, supplier: np.ndarray) -> np.ndarray:
    """Align supplier onto master using ORB feature matching."""
    supplier_rs = _resize_match(master, supplier)
    gray_m = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(supplier_rs, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(gray_m, None)
    kp2, des2 = orb.detectAndCompute(gray_s, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return supplier_rs

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    except Exception:
        return supplier_rs

    if len(good) < 10:
        return supplier_rs

    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        return supplier_rs

    h, w = master.shape[:2]
    return cv2.warpPerspective(supplier_rs, H, (w, h),
                               flags=cv2.INTER_LANCZOS4,
                               borderValue=(255,255,255))

def compute_diff(master: np.ndarray, supplier: np.ndarray,
                 threshold: int = 20) -> DiffResult:
    """Pixel-by-pixel comparison and heatmap generation."""
    if master.shape != supplier.shape:
        supplier = _resize_match(master, supplier)

    diff = cv2.absdiff(master, supplier)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    diff_dilated = cv2.dilate(diff_thresh, kernel, iterations=2)

    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    mask_inv = cv2.bitwise_not(diff_dilated)
    heatmap[mask_inv > 0] = [255, 255, 255]

    overlay = cv2.addWeighted(master, 0.55, heatmap, 0.45, 0)
    diff_3ch = cv2.cvtColor(diff_dilated, cv2.COLOR_GRAY2BGR)
    overlay = np.where(diff_3ch > 0, heatmap, overlay).astype(np.uint8)

    diff_px = int(np.count_nonzero(diff_dilated))
    total = diff_dilated.size

    return DiffResult(
        heatmap_bgr=heatmap,
        overlay_bgr=overlay,
        aligned_supplier=supplier,
        diff_score=diff_px / total if total > 0 else 0.0,
        diff_pixel_count=diff_px,
        total_pixels=total,
    )
