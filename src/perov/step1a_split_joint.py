#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_pdf_and_stitch_md.py

Usage:
  1) Split PDF into single-page PDFs:
     python split_pdf_and_stitch_md.py split input.pdf --out_dir pages/

  2) After you convert each pages/page_XXXX.pdf -> md (any tool),
     stitch them into one MD with guaranteed PAGE markers:
     python split_pdf_and_stitch_md.py stitch pages/ --md_glob "*.md" --out_md paper.raw.md

Notes:
- Page numbering in output uses 0-based indexing: [[PAGE:0:0]], [[PAGE:1:0]], ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader, PdfWriter
except ImportError as e:
    raise SystemExit("Install pypdf: pip install pypdf") from e


def split_pdf_to_pages(pdf_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    out_files: List[Path] = []

    for i, page in enumerate(reader.pages):
        w = PdfWriter()
        w.add_page(page)
        out = out_dir / f"page_{i:04d}.pdf"
        with out.open("wb") as f:
            w.write(f)
        out_files.append(out)

    return out_files


def stitch_md_pages(md_dir: Path, md_glob: str, out_md: Path) -> None:
    mds = sorted(md_dir.glob(md_glob))
    if not mds:
        raise FileNotFoundError(f"No md files matched {md_glob} in {md_dir}")

    parts: List[str] = []
    for i, md_path in enumerate(mds):
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        parts.append(f"[[PAGE:{i}:0]]\n\n{text}\n")

    out_md.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_split = sub.add_parser("split")
    ap_split.add_argument("pdf", type=str)
    ap_split.add_argument("--out_dir", type=str, required=True)

    ap_stitch = sub.add_parser("stitch")
    ap_stitch.add_argument("md_dir", type=str)
    ap_stitch.add_argument("--md_glob", type=str, default="page_*.md")
    ap_stitch.add_argument("--out_md", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "split":
        pdf = Path(args.pdf)
        out_dir = Path(args.out_dir)
        files = split_pdf_to_pages(pdf, out_dir)
        print(f"[OK] split into {len(files)} pages at: {out_dir}")
        return 0

    if args.cmd == "stitch":
        md_dir = Path(args.md_dir)
        out_md = Path(args.out_md)
        stitch_md_pages(md_dir, args.md_glob, out_md)
        print(f"[OK] wrote: {out_md}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
