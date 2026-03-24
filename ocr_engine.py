"""
ocr_engine.py — extracts and compares text using EasyOCR
EasyOCR is lighter than PaddleOCR and works great on Streamlit Cloud free tier.
"""
from __future__ import annotations
import re
import difflib
from dataclasses import dataclass
from PIL import Image
import numpy as np

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# Cache the reader so it only loads once
_reader_cache: dict = {}

def _get_reader(langs=('en',)):
    key = str(langs)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(list(langs), gpu=False)
    return _reader_cache[key]

@dataclass
class TextDiscrepancy:
    line_index: int
    master_text: str
    supplier_text: str
    similarity: float
    category: str   # CRITICAL / WARNING / INFO
    reason: str

@dataclass
class OcrResult:
    master_lines: list
    supplier_lines: list
    discrepancies: list
    overall_similarity: float

_PATTERNS = {
    "dimension":    re.compile(r"\d+\s*(?:mm|cm|in)", re.I),
    "chemical":     re.compile(r"\d+\s*%"),
    "product_ref":  re.compile(r"(?:REF|LOT|AQL|REV)[:\s#]*[\w\-\.]+", re.I),
    "regulation":   re.compile(r"(?:EU|ISO|EN)\s*[\d/]+", re.I),
    "email":        re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I),
    "date":         re.compile(r"\b\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}\b"),
}

def _classify(master: str, supplier: str):
    for key, pat in _PATTERNS.items():
        if pat.search(master) or pat.search(supplier):
            return "CRITICAL", f"Mismatch in {key.replace('_',' ')} field"
    clean_m = re.sub(r"\s+", " ", master.strip()).lower()
    clean_s = re.sub(r"\s+", " ", supplier.strip()).lower()
    if clean_m == clean_s:
        return "INFO", "Case or whitespace difference only"
    return "WARNING", "General text mismatch"

def extract_text(pil_pages: list) -> list:
    if not _OCR_AVAILABLE:
        raise RuntimeError("EasyOCR not installed. Run: pip install easyocr")
    reader = _get_reader(('en', 'ch_sim'))
    lines = []
    for page in pil_pages:
        arr = np.array(page)
        results = reader.readtext(arr, detail=0, paragraph=False)
        lines.extend([r.strip() for r in results if r.strip()])
    return lines

def compare_texts(master_lines: list, supplier_lines: list) -> OcrResult:
    discrepancies = []
    matcher = difflib.SequenceMatcher(None, master_lines, supplier_lines)
    all_sims = []
    idx = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for _ in master_lines[i1:i2]:
                all_sims.append(1.0)
                idx += 1

        elif tag == "replace":
            mc = master_lines[i1:i2]
            sc = supplier_lines[j1:j2]
            ml = max(len(mc), len(sc))
            mc += [""] * (ml - len(mc))
            sc += [""] * (ml - len(sc))
            for m, s in zip(mc, sc):
                sim = difflib.SequenceMatcher(None, m, s).ratio()
                all_sims.append(sim)
                if sim < 0.98:
                    cat, reason = _classify(m, s)
                    discrepancies.append(TextDiscrepancy(idx, m, s, round(sim,4), cat, reason))
                idx += 1

        elif tag == "delete":
            for m in master_lines[i1:i2]:
                all_sims.append(0.0)
                discrepancies.append(TextDiscrepancy(idx, m,
                    "⚠ MISSING IN SUPPLIER", 0.0, "CRITICAL",
                    "Text in master but absent from supplier proof"))
                idx += 1

        elif tag == "insert":
            for s in supplier_lines[j1:j2]:
                all_sims.append(0.0)
                discrepancies.append(TextDiscrepancy(idx,
                    "⚠ NOT IN MASTER", s, 0.0, "WARNING",
                    "Extra text in supplier proof not in master"))
                idx += 1

    overall = sum(all_sims)/len(all_sims) if all_sims else 1.0
    return OcrResult(master_lines, supplier_lines, discrepancies, round(overall,4))
