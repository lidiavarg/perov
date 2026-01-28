#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def run_stream(cmd: list[str]) -> int:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    return proc.wait()


def find_single_md(marker_out: Path) -> Path | None:
    mds = sorted(marker_out.rglob("*.md"))
    return mds[0] if mds else None


def guess_page_index(pdf_path: Path) -> int:
    # expects page_0007.pdf -> 7
    m = re.search(r"(\d+)", pdf_path.stem)
    if not m:
        raise ValueError(f"Cannot infer page number from: {pdf_path.name}")
    return int(m.group(1))


def copy_images(marker_out_dir: Path, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in marker_out_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(marker_out_dir)
            safe_name = "__".join(rel.parts)
            dst = images_dir / safe_name
            shutil.copy2(p, dst)
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True, help="Working dir for marker output (can be temp)")
    ap.add_argument("--md_out_dir", type=Path, required=True, help="Where to write page_XXXX.md")
    ap.add_argument("--force_ocr", action="store_true")
    ap.add_argument("--disable_image_extraction", action="store_true")
    ap.add_argument("--redo_inline_math", action="store_true")

    ap.add_argument("--extract_images", action="store_true")
    ap.add_argument("--images_dir", type=Path, default=None)

    ap.add_argument("--marker_llm", action="store_true")
    ap.add_argument("--ollama_base_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama_model", type=str, default="llama3.1")
    ap.add_argument("--save_marker_json", action="store_true")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.md_out_dir.mkdir(parents=True, exist_ok=True)

    page_i = guess_page_index(args.pdf)
    out_md = args.md_out_dir / f"page_{page_i:04d}.md"

    flags: list[str] = []
    if args.force_ocr:
        flags.append("--force_ocr")
    if args.disable_image_extraction:
        flags.append("--disable_image_extraction")
    if args.redo_inline_math:
        flags.append("--redo_inline_math")

    if args.marker_llm:
        flags += [
            "--use_llm",
            "--llm_service=marker.services.ollama.OllamaService",
            "--ollama_base_url", args.ollama_base_url,
            "--ollama_model", args.ollama_model,
        ]

    t0 = time.time()

    # markdown
    rc = run_stream(
        [
            "marker_single",
            str(args.pdf),
            "--output_format", "markdown",
            "--output_dir", str(args.out_dir),
            *flags,
        ]
    )
    if rc != 0:
        raise SystemExit(rc)

    md_path = find_single_md(args.out_dir)
    if not md_path:
        raise FileNotFoundError(f"No markdown produced in: {args.out_dir}")

    text = md_path.read_text(encoding="utf-8", errors="ignore").strip() + "\n"
    out_md.write_text(text, encoding="utf-8")
    print(f"[OK] wrote page md: {out_md}")

    # marker json 
    if args.save_marker_json:
        rc = run_stream(
            [
                "marker_single",
                str(args.pdf),
                "--output_format", "json",
                "--output_dir", str(args.out_dir),
                *flags,
            ]
        )
        if rc != 0:
            raise SystemExit(rc)

    # images 
    if args.extract_images:
        if not args.images_dir:
            raise SystemExit("--extract_images требует --images_dir")
        n = copy_images(args.out_dir, args.images_dir)
        print(f"[OK] copied images: {n} -> {args.images_dir}")

    dt = time.time() - t0
    print(f"[OK] Marker page {page_i} done in {dt:.1f}s")


if __name__ == "__main__":
    main()

