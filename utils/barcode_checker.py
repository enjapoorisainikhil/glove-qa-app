"""
barcode_checker.py — scans and compares barcodes in both files
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass

try:
    from pyzbar import pyzbar
    _PYZBAR_OK = True
except ImportError:
    _PYZBAR_OK = False

@dataclass
class BarcodeInfo:
    data: str
    barcode_type: str

@dataclass
class BarcodeCheckResult:
    master_barcodes: list
    supplier_barcodes: list
    matched: list
    missing_in_supplier: list
    extra_in_supplier: list
    mismatched: list
    passed: bool

def _preprocess(img: np.ndarray) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = img.shape[:2]
    scale = max(1, 2000 // max(h, w))
    up = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC) if scale > 1 else gray
    return [gray, binary, up]

def scan_barcodes(img: np.ndarray) -> list:
    if not _PYZBAR_OK:
        raise RuntimeError("pyzbar not installed")
    found = {}
    for variant in _preprocess(img):
        for d in pyzbar.decode(variant):
            data = d.data.decode("utf-8", errors="replace").strip()
            key = f"{d.type}:{data}"
            if key not in found:
                found[key] = BarcodeInfo(data=data, barcode_type=d.type)
    return list(found.values())

def verify_barcodes(master_img: np.ndarray, supplier_img: np.ndarray) -> BarcodeCheckResult:
    master_bc = scan_barcodes(master_img)
    supplier_bc = scan_barcodes(supplier_img)

    master_map = {b.data: b for b in master_bc}
    supplier_map = {b.data: b for b in supplier_bc}

    matched, missing, extra, mismatched = [], [], [], []

    for data, mb in master_map.items():
        if data in supplier_map:
            matched.append((mb, supplier_map[data]))
        else:
            same_type = [s for s in supplier_bc if s.barcode_type == mb.barcode_type]
            if same_type:
                mismatched.append((mb, same_type[0]))
            else:
                missing.append(mb)

    for data, sb in supplier_map.items():
        if data not in master_map:
            extra.append(sb)

    passed = len(missing) == 0 and len(mismatched) == 0

    return BarcodeCheckResult(
        master_barcodes=master_bc,
        supplier_barcodes=supplier_bc,
        matched=matched,
        missing_in_supplier=missing,
        extra_in_supplier=extra,
        mismatched=mismatched,
        passed=passed,
    )
