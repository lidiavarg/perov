#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF


# -------- PDF text --------
def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)


# -------- Normalization (robust to wrapping/reflow) --------
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    # Keep latin+digits+basic math + greek range (often in papers)
    s = re.sub(r"[^a-z0-9α-ω=+\-*/().,% ]", "", s)
    return s.strip()


def token_set(s: str) -> set[str]:
    return set(s.split())


# -------- Metrics --------
def precision_recall(pdf_norm: str, md_norm: str) -> tuple[float, float]:
    pdf_t = token_set(pdf_norm)
    md_t = token_set(md_norm)
    inter = pdf_t & md_t
    precision = len(inter) / max(1, len(md_t))  # "MD tokens supported by PDF"
    recall = len(inter) / max(1, len(pdf_t))    # "PDF tokens preserved in MD"
    return precision, recall


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def structure_score(md_raw: str) -> float:
    # very simple “did we keep structure” signals
    has_headings = bool(re.search(r"^#{1,6}\s+\S", md_raw, re.M))
    has_math = bool(re.search(r"(?<!\\)\$[^$]+\$", md_raw) or re.search(r"\$\$[\s\S]+?\$\$", md_raw))
    has_tables = "|" in md_raw
    has_figures = bool(re.search(r"!\[.*?\]\(.*?\)", md_raw))
    return (has_headings + has_math + has_tables + has_figures) / 4.0


def noise_score(md_raw: str) -> float:
    # control chars except newline/tab are usually extraction garbage
    bad = 0
    for ch in md_raw:
        if ch in "\n\t\r":
            continue
        if unicodedata.category(ch).startswith("C"):
            bad += 1
    # also penalize replacement char �
    bad += md_raw.count("\uFFFD")
    return max(0.0, 1.0 - bad / max(1, len(md_raw)))


def main():
    ap = argparse.ArgumentParser(description="Compare existing Markdown against source PDF")
    ap.add_argument("--pdf", required=True, help="Path to source PDF")
    ap.add_argument("--md", required=True, help="Path to existing extracted MD")
    ap.add_argument("--out", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    md_path = Path(args.md)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"MD not found: {md_path}")

    pdf_raw = extract_pdf_text(pdf_path)
    md_raw = md_path.read_text(encoding="utf-8", errors="replace")

    pdf_norm = normalize(pdf_raw)
    md_norm = normalize(md_raw)

    p, r = precision_recall(pdf_norm, md_norm)

    result = {
        "pdf": str(pdf_path),
        "md": str(md_path),
        "text_precision": round(p, 3),
        "text_recall": round(r, 3),
        "f1_accuracy": round(f1(p, r), 3),
        "structure_score": round(structure_score(md_raw), 3),
        "noise_score": round(noise_score(md_raw), 3),
    }
    result["overall_accuracy"] = round(
        0.50 * result["f1_accuracy"] +
        0.25 * result["structure_score"] +
        0.25 * result["noise_score"],
        3
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
