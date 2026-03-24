"""
pdf_parser.py — converts PDF files into images for analysis
"""
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import cv2

DPI = 200

def pdf_bytes_to_pil(file_bytes: bytes) -> list:
    """Convert PDF bytes to list of PIL images."""
    return convert_from_bytes(file_bytes, dpi=DPI, fmt="RGB")

def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR image."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def autocrop(img: np.ndarray, padding: int = 15) -> np.ndarray:
    """Remove white borders around artwork."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    H, W = img.shape[:2]
    return img[max(0,y-padding):min(H,y+h+padding),
               max(0,x-padding):min(W,x+w+padding)]

def load_document(file_bytes: bytes):
    """
    Main entry point.
    Returns (cv_image, pil_pages)
    """
    pil_pages = pdf_bytes_to_pil(file_bytes)
    # Stack all pages vertically into one image
    cv_pages = [pil_to_cv(p) for p in pil_pages]
    if len(cv_pages) == 1:
        composite = cv_pages[0]
    else:
        max_w = max(p.shape[1] for p in cv_pages)
        padded = []
        for p in cv_pages:
            h, w = p.shape[:2]
            if w < max_w:
                pad = np.full((h, max_w - w, 3), 255, dtype=np.uint8)
                p = np.hstack([p, pad])
            padded.append(p)
        composite = np.vstack(padded)
    cropped = autocrop(composite)
    return cropped, pil_pages
