#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader, PdfWriter
except ImportError as e:
    raise SystemExit("Install pypdf: pip install pypdf") from e


# Matches:
#   ![](_page_0_Figure_12.jpeg)
#   ![alt text](../imgs/_page_0_Figure_12.jpeg)
IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# Matches:
#   [[ASSET: page=0, id=Figure_12, file=_page_0_Figure_12.jpeg]]
ASSET_RE = re.compile(
    r"\[\[ASSET:\s*page=(\d+),\s*id=([^,]+),\s*file=([^\]]+)\]\]"
)

FILE_PAGE_RE = re.compile(r"_page_(\d+)_")


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


def _rewrite_page0_filename(path_str: str, page_idx: int) -> str:
    """
    Rewrite any *_page_0_* to *_page_{page_idx}_*.
    Works even if path includes directories.
    """
    if "_page_0_" not in path_str:
        return path_str
    return FILE_PAGE_RE.sub(f"_page_{page_idx}_", path_str, count=1)


def _maybe_rename_image(images_dir: Path | None, old_path: str, new_path: str) -> None:
    """
    Rename only by basename in images_dir (common case: all images copied to one folder).
    Safe if file doesn't exist.
    """
    if not images_dir:
        return
    src = images_dir / Path(old_path).name
    dst = images_dir / Path(new_path).name
    if src.exists() and not dst.exists():
        src.rename(dst)


def rewrite_images_and_assets_for_page(
    text: str,
    page_idx: int,
    images_dir: Path | None,
    rename_images: bool,
) -> str:
    # 1) Rewrite markdown images ![](...)
    def img_repl(m: re.Match) -> str:
        old = m.group(1).strip()
        new = _rewrite_page0_filename(old, page_idx)
        if rename_images and new != old:
            _maybe_rename_image(images_dir, old, new)
        return m.group(0).replace(old, new)

    text = IMG_MD_RE.sub(img_repl, text)

    # 2) Rewrite already-normalized ASSET markers (если они есть)
    def asset_repl(m: re.Match) -> str:
        old_page = int(m.group(1))
        asset_id = m.group(2).strip()
        file_path = m.group(3).strip()

        # В per-page marker обычно page=0 — правим на реальный i
        new_file = _rewrite_page0_filename(file_path, page_idx)
        new_page = page_idx if old_page == 0 else old_page

        if rename_images and new_file != file_path:
            _maybe_rename_image(images_dir, file_path, new_file)

        return f"[[ASSET: page={new_page}, id={asset_id}, file={new_file}]]"

    text = ASSET_RE.sub(asset_repl, text)
    return text


def stitch_md_pages(
    md_dir: Path,
    md_glob: str,
    out_md: Path,
    images_dir: Path | None = None,
    rename_images: bool = False,
) -> None:
    mds = sorted(md_dir.glob(md_glob))
    if not mds:
        raise FileNotFoundError(f"No md files matched {md_glob} in {md_dir}")

    parts: List[str] = []
    for i, md_path in enumerate(mds):
        raw = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        fixed = rewrite_images_and_assets_for_page(
            raw,
            page_idx=i,
            images_dir=images_dir,
            rename_images=rename_images,
        )
        parts.append(f"[[PAGE:{i}:0]]\n\n{fixed}\n")

    out_md.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_split = sub.add_parser("split")
    ap_split.add_argument("pdf", type=Path)
    ap_split.add_argument("--out_dir", type=Path, required=True)

    ap_stitch = sub.add_parser("stitch")
    ap_stitch.add_argument("md_dir", type=Path)
    ap_stitch.add_argument("--md_glob", type=str, default="page_*.md")
    ap_stitch.add_argument("--out_md", type=Path, required=True)
    ap_stitch.add_argument("--images_dir", type=Path, default=None)
    ap_stitch.add_argument("--rename_images", action="store_true")

    args = ap.parse_args()

    if args.cmd == "split":
        files = split_pdf_to_pages(args.pdf, args.out_dir)
        print(f"[OK] split into {len(files)} pages at: {args.out_dir}")
        return 0

    if args.cmd == "stitch":
        stitch_md_pages(
            md_dir=args.md_dir,
            md_glob=args.md_glob,
            out_md=args.out_md,
            images_dir=args.images_dir,
            rename_images=args.rename_images,
        )
        print(f"[OK] wrote: {args.out_md}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
