"""
GloveQA — Automated Artwork Proof Verification
Streamlit Cloud ready. Uses EasyOCR (lightweight, no GPU needed).
"""
from __future__ import annotations
import io, json, datetime, traceback
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.pdf_parser import load_document
from utils.visual_diff import align_images, compute_diff
from utils.barcode_checker import verify_barcodes

try:
    from utils.ocr_engine import extract_text, compare_texts, OcrResult
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GloveQA · Artwork Verifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root{
  --bg:#0D0F14;--surface:#14171F;--surface2:#1C2030;--border:#262B3A;
  --accent:#00FFB2;--red:#FF3C6E;--amber:#FFD166;
  --text:#E8ECF4;--dim:#6B7490;
  --mono:'Space Mono',monospace;--sans:'Syne',sans-serif;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem 4rem!important;max-width:1400px!important;}

.hdr{display:flex;align-items:center;gap:1rem;margin-bottom:.5rem;}
.logo{font-family:var(--mono);font-size:2.8rem;color:var(--accent);line-height:1;}
.title{font-size:2rem;font-weight:800;letter-spacing:-.03em;line-height:1;}
.sub{font-family:var(--mono);font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);margin-top:.3rem;}

.stButton>button{
  width:100%;background:var(--accent)!important;color:#071A12!important;
  font-family:var(--mono)!important;font-weight:700!important;font-size:1rem!important;
  letter-spacing:.08em!important;border:none!important;border-radius:8px!important;
  padding:.9rem 2rem!important;
}
.stButton>button:hover{opacity:.85!important;}

.pass{background:linear-gradient(90deg,#00FFB2,#00C98A);color:#071A12;border-radius:10px;
  padding:1.2rem 2rem;font-family:var(--mono);font-size:1.5rem;font-weight:700;
  letter-spacing:.06em;text-align:center;margin:1rem 0;}
.fail{background:linear-gradient(90deg,#FF3C6E,#CC1F4F);color:#fff;border-radius:10px;
  padding:1.2rem 2rem;font-family:var(--mono);font-size:1.5rem;font-weight:700;
  letter-spacing:.06em;text-align:center;margin:1rem 0;}
.warn{background:linear-gradient(90deg,#FFD166,#E6A820);color:#1A1000;border-radius:10px;
  padding:1.2rem 2rem;font-family:var(--mono);font-size:1.5rem;font-weight:700;
  letter-spacing:.06em;text-align:center;margin:1rem 0;}

.sec{font-family:var(--mono);font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;
  color:var(--accent);border-bottom:1px solid var(--border);padding-bottom:.4rem;margin:2rem 0 1rem;}

.mrow{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.2rem;}
.mc{flex:1;min-width:130px;background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:1rem 1.2rem;}
.ml{font-family:var(--mono);font-size:.6rem;color:var(--dim);letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:.3rem;}
.mv{font-family:var(--mono);font-size:1.5rem;font-weight:700;}
.mv.g{color:var(--accent);}.mv.b{color:var(--red);}.mv.w{color:var(--amber);}

.bc{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:.8rem 1.2rem;margin-bottom:.6rem;font-family:var(--mono);font-size:.78rem;}
.bc.ok{border-left:4px solid var(--accent);}
.bc.fail{border-left:4px solid var(--red);}
.bc.warn{border-left:4px solid var(--amber);}

[data-testid="stFileUploader"]{background:var(--surface)!important;
  border:1.5px dashed var(--border)!important;border-radius:10px!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:8px!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;}
.stProgress>div>div{background-color:var(--accent)!important;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="logo">⬡</div>
  <div>
    <div class="title">GloveQA</div>
    <div class="sub">Automated Artwork Proof Verification · Supply Chain QA</div>
  </div>
</div>
""", unsafe_allow_html=True)

ocr_badge = "🟢 OCR Ready (EasyOCR)" if OCR_AVAILABLE else "🔴 EasyOCR not found — text check disabled"
st.caption(ocr_badge)
st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

# ── Upload zone ───────────────────────────────────────────────────────────────
col1, _, col2 = st.columns([5, .4, 5])
with col1:
    st.markdown('<div style="font-family:monospace;font-size:.7rem;letter-spacing:.15em;color:#00FFB2;text-transform:uppercase;margin-bottom:.3rem">① Customer Master Artwork</div>', unsafe_allow_html=True)
    master_file = st.file_uploader("Master", type=["pdf","ai"], label_visibility="collapsed", key="master")
with col2:
    st.markdown('<div style="font-family:monospace;font-size:.7rem;letter-spacing:.15em;color:#00FFB2;text-transform:uppercase;margin-bottom:.3rem">② Supplier Proof File</div>', unsafe_allow_html=True)
    supplier_file = st.file_uploader("Supplier", type=["pdf","ai"], label_visibility="collapsed", key="supplier")

# ── Thumbnails ────────────────────────────────────────────────────────────────
if master_file or supplier_file:
    t1, _, t2 = st.columns([5, .4, 5])
    if master_file:
        with t1:
            try:
                master_file.seek(0)
                img = Image.open(io.BytesIO(master_file.read()))
                st.image(img, caption="Master preview", use_container_width=True)
                master_file.seek(0)
            except Exception:
                master_file.seek(0)
                st.info("PDF uploaded ✓")
    if supplier_file:
        with t2:
            try:
                supplier_file.seek(0)
                img = Image.open(io.BytesIO(supplier_file.read()))
                st.image(img, caption="Supplier preview", use_container_width=True)
                supplier_file.seek(0)
            except Exception:
                supplier_file.seek(0)
                st.info("PDF uploaded ✓")

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 4, 2])
with btn_col:
    run = st.button("🔬  ANALYZE PROOF", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def bgr2pil(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil2bytes(img):
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()
def mc(label, value, cls=""):
    return f'<div class="mc"><div class="ml">{label}</div><div class="mv {cls}">{value}</div></div>'
def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
if run:
    if not master_file or not supplier_file:
        st.error("⚠ Please upload BOTH files before analysing.")
        st.stop()

    overall_pass = True
    warn_count   = 0
    errors       = []

    prog = st.progress(0, text="Starting analysis…")

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    sec("STEP 1 · Document Parsing & Normalisation")
    try:
        master_bytes   = master_file.read()
        supplier_bytes = supplier_file.read()
        with st.spinner("Rasterising master…"):
            master_cv, master_pil = load_document(master_bytes)
        prog.progress(20, text="Master parsed ✓")
        with st.spinner("Rasterising supplier…"):
            supplier_cv, supplier_pil = load_document(supplier_bytes)
        prog.progress(35, text="Supplier parsed ✓")

        c1, c2 = st.columns(2)
        with c1: st.image(bgr2pil(master_cv),   caption=f"Master — {master_cv.shape[1]}×{master_cv.shape[0]}px",   use_container_width=True)
        with c2: st.image(bgr2pil(supplier_cv), caption=f"Supplier — {supplier_cv.shape[1]}×{supplier_cv.shape[0]}px", use_container_width=True)
    except Exception as e:
        st.error(f"Parsing failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

    # ── Step 2: Visual Diff ───────────────────────────────────────────────────
    sec("STEP 2 · Visual Alignment & Pixel Diff")
    diff_pct = 0.0
    try:
        with st.spinner("Aligning images…"):
            aligned = align_images(master_cv, supplier_cv)
        prog.progress(52, text="Alignment done ✓")
        with st.spinner("Computing pixel diff…"):
            dr = compute_diff(master_cv, aligned)
        prog.progress(65, text="Diff computed ✓")

        diff_pct    = dr.diff_score * 100
        visual_pass = diff_pct < 2.0
        if not visual_pass:
            overall_pass = False
            errors.append(f"Visual diff: {diff_pct:.2f}% pixels differ (limit 2%)")
        elif diff_pct > 0.5:
            warn_count += 1

        clr = "g" if visual_pass else "b"
        st.markdown(
            '<div class="mrow">'
            + mc("Diff Score",      f"{diff_pct:.2f}%",           clr)
            + mc("Changed Pixels",  f"{dr.diff_pixel_count:,}",   clr)
            + mc("Total Pixels",    f"{dr.total_pixels:,}")
            + mc("Visual QA",       "PASS ✓" if visual_pass else "FAIL ✗", clr)
            + '</div>', unsafe_allow_html=True)

        h1, h2 = st.columns(2)
        with h1: st.image(bgr2pil(dr.overlay_bgr),      caption="🔴 Heatmap — red = changed",  use_container_width=True)
        with h2: st.image(bgr2pil(dr.aligned_supplier), caption="Aligned Supplier Proof",        use_container_width=True)
        st.download_button("⬇ Download Heatmap", pil2bytes(bgr2pil(dr.overlay_bgr)), "heatmap.png", "image/png")

    except Exception as e:
        st.warning(f"Visual diff error: {e}")
        warn_count += 1

    # ── Step 3: OCR ───────────────────────────────────────────────────────────
    sec("STEP 3 · Deep OCR & Text Verification")
    ocr_result: Optional[OcrResult] = None
    if not OCR_AVAILABLE:
        st.warning("EasyOCR not installed — text check skipped.")
        warn_count += 1
    else:
        try:
            with st.spinner("Extracting text from master (this takes ~30 seconds first run)…"):
                master_lines = extract_text(master_pil)
            prog.progress(78, text="Master OCR done ✓")
            with st.spinner("Extracting text from supplier…"):
                supplier_lines = extract_text(supplier_pil)
            prog.progress(88, text="Supplier OCR done ✓")

            ocr_result = compare_texts(master_lines, supplier_lines)
            crit  = [d for d in ocr_result.discrepancies if d.category == "CRITICAL"]
            warns = [d for d in ocr_result.discrepancies if d.category == "WARNING"]

            if crit:
                overall_pass = False
                errors.append(f"OCR: {len(crit)} critical text discrepanc{'y' if len(crit)==1 else 'ies'}")
            warn_count += len(warns)

            sim = ocr_result.overall_similarity * 100
            oc  = "g" if not crit else "b"
            st.markdown(
                '<div class="mrow">'
                + mc("Text Similarity",  f"{sim:.1f}%",              oc)
                + mc("Critical Issues",  str(len(crit)),  "b" if crit  else "g")
                + mc("Warnings",         str(len(warns)), "w" if warns else "g")
                + mc("Master Lines",     str(len(master_lines)))
                + mc("Supplier Lines",   str(len(supplier_lines)))
                + '</div>', unsafe_allow_html=True)

            if ocr_result.discrepancies:
                rows = []
                for d in ocr_result.discrepancies:
                    e = {"CRITICAL":"🔴","WARNING":"🟡","INFO":"⚪"}.get(d.category,"")
                    rows.append({"Severity":f"{e} {d.category}","Reason":d.reason,
                                 "Master Text":d.master_text[:120],
                                 "Supplier Text":d.supplier_text[:120],
                                 "Similarity":f"{d.similarity*100:.1f}%"})
                df = pd.DataFrame(rows)
                with st.expander(f"📋 {len(rows)} Text Discrepanc{'y' if len(rows)==1 else 'ies'} Found", expanded=True):
                    st.dataframe(df, use_container_width=True,
                                 column_config={"Master Text":st.column_config.TextColumn(width="large"),
                                                "Supplier Text":st.column_config.TextColumn(width="large")})
                    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), "text_report.csv", "text/csv")
            else:
                st.success("✅ No text discrepancies — all lines match!")
        except Exception as e:
            st.warning(f"OCR error: {e}")
            st.code(traceback.format_exc())
            warn_count += 1

    # ── Step 4: Barcodes ──────────────────────────────────────────────────────
    sec("STEP 4 · Barcode & Data Matrix Integrity Check")
    bc_result = None
    try:
        with st.spinner("Scanning barcodes…"):
            bc_result = verify_barcodes(master_cv, supplier_cv)
        prog.progress(96, text="Barcode scan done ✓")

        if not bc_result.passed:
            overall_pass = False
            errors.append(f"Barcode FAIL: {len(bc_result.missing_in_supplier)} missing, {len(bc_result.mismatched)} mismatched")

        bc_clr = "g" if bc_result.passed else "b"
        st.markdown(
            '<div class="mrow">'
            + mc("Master Barcodes",    str(len(bc_result.master_barcodes)))
            + mc("Supplier Barcodes",  str(len(bc_result.supplier_barcodes)))
            + mc("Matched",            str(len(bc_result.matched)),            "g" if bc_result.matched else "")
            + mc("Missing in Proof",   str(len(bc_result.missing_in_supplier)), "b" if bc_result.missing_in_supplier else "g")
            + mc("Mismatched",         str(len(bc_result.mismatched)),          "b" if bc_result.mismatched else "g")
            + '</div>', unsafe_allow_html=True)

        if bc_result.matched:
            with st.expander(f"✅ {len(bc_result.matched)} Matched Barcode(s)"):
                for m, s in bc_result.matched:
                    st.markdown(f'<div class="bc ok">✅ <b>MATCH</b> &nbsp;|&nbsp; <code>{m.data}</code> &nbsp;|&nbsp; {m.barcode_type}</div>', unsafe_allow_html=True)
        if bc_result.mismatched:
            with st.expander(f"🔴 {len(bc_result.mismatched)} MISMATCHED — CRITICAL", expanded=True):
                for m, s in bc_result.mismatched:
                    st.markdown(f'<div class="bc fail">MASTER: <code>{m.data}</code><br>SUPPLIER: <code>{s.data}</code></div>', unsafe_allow_html=True)
        if bc_result.missing_in_supplier:
            with st.expander(f"🔴 {len(bc_result.missing_in_supplier)} Missing From Proof", expanded=True):
                for b in bc_result.missing_in_supplier:
                    st.markdown(f'<div class="bc fail">MISSING: <code>{b.data}</code> ({b.barcode_type})</div>', unsafe_allow_html=True)
        if bc_result.extra_in_supplier:
            with st.expander(f"🟡 {len(bc_result.extra_in_supplier)} Extra in Proof"):
                for b in bc_result.extra_in_supplier:
                    st.markdown(f'<div class="bc warn">EXTRA: <code>{b.data}</code> ({b.barcode_type})</div>', unsafe_allow_html=True)
        if not bc_result.master_barcodes:
            st.info("ℹ No barcodes detected — may be expected for this artwork type.")
    except Exception as e:
        st.warning(f"Barcode scan skipped: {e}")
        warn_count += 1

    prog.progress(100, text="Analysis complete ✓")

    # ── Final Verdict ─────────────────────────────────────────────────────────
    sec("FINAL QA VERDICT")

    if overall_pass and warn_count == 0:
        st.markdown('<div class="pass">✅ &nbsp; PASS — PROOF APPROVED</div>', unsafe_allow_html=True)
        st.balloons()
    elif overall_pass:
        st.markdown(f'<div class="warn">⚠ &nbsp; CONDITIONAL PASS — {warn_count} WARNING(S) — REVIEW REQUIRED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fail">✖ &nbsp; FAIL — REJECT PROOF</div>', unsafe_allow_html=True)
        for err in errors:
            st.markdown(f"- 🔴 {err}")

    # Summary table
    summary = {"Check":["Parsing","Visual Diff","OCR Text","Barcodes"],
               "Status":[], "Detail":[]}
    summary["Status"].append("✅ PASS")
    summary["Detail"].append(f"Master {master_cv.shape[1]}×{master_cv.shape[0]}px")
    summary["Status"].append("✅ PASS" if diff_pct < 2.0 else "✖ FAIL")
    summary["Detail"].append(f"{diff_pct:.2f}% pixels differ")
    if ocr_result:
        crit_n = len([d for d in ocr_result.discrepancies if d.category=="CRITICAL"])
        summary["Status"].append("✅ PASS" if crit_n==0 else "✖ FAIL")
        summary["Detail"].append(f"{len(ocr_result.discrepancies)} issues · {ocr_result.overall_similarity*100:.1f}% similarity")
    else:
        summary["Status"].append("⚠ SKIPPED"); summary["Detail"].append("EasyOCR not installed")
    if bc_result:
        summary["Status"].append("✅ PASS" if bc_result.passed else "✖ FAIL")
        summary["Detail"].append(f"{len(bc_result.matched)} matched · {len(bc_result.mismatched)} mismatched")
    else:
        summary["Status"].append("⚠ SKIPPED"); summary["Detail"].append("pyzbar not installed")

    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    # JSON export
    report = {
        "generated_at": datetime.datetime.utcnow().isoformat()+"Z",
        "verdict": "PASS" if (overall_pass and warn_count==0) else ("CONDITIONAL" if overall_pass else "FAIL"),
        "errors": errors,
        "visual_diff_pct": round(diff_pct, 4),
        "ocr_discrepancies": [{"line":d.line_index,"cat":d.category,"reason":d.reason,
                                "master":d.master_text,"supplier":d.supplier_text}
                               for d in (ocr_result.discrepancies if ocr_result else [])],
        "barcodes": {
            "master":   [{"data":b.data,"type":b.barcode_type} for b in (bc_result.master_barcodes   if bc_result else [])],
            "supplier": [{"data":b.data,"type":b.barcode_type} for b in (bc_result.supplier_barcodes if bc_result else [])],
        }
    }
    st.download_button(
        "⬇ Download Full QA Report (JSON)",
        json.dumps(report, indent=2, ensure_ascii=False).encode(),
        f"gloveqa_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json",
    )
