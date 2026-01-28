
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_py(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({proc.returncode}): {' '.join(cmd)}")


def ensure_ollama_up(ollama_base_url: str) -> None:
    import urllib.request
    url = ollama_base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=3) as r:  # nosec
        if r.status != 200:
            raise RuntimeError(f"Ollama /api/tags returned status {r.status}")


def list_page_pdfs(pages_dir: Path) -> list[Path]:
    pdfs = sorted(pages_dir.glob("page_*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No page PDFs found in: {pages_dir}")
    return pdfs


# OCR probe
OcrMode = Literal["none", "strip_existing_ocr", "force_ocr"]


@dataclass(frozen=True)
class PdfTextProbe:
    chars: int
    letters: int
    spaces: int
    replacement: int
    control: int
    sample: str


def probe_pdf_text(pdf: Path, pages: int = 2) -> PdfTextProbe:
    try:
        import fitz  # type: ignore
    except Exception as e:
        s = f"[probe unavailable: {e}]"
        return PdfTextProbe(chars=len(s), letters=0, spaces=0, replacement=0, control=0, sample=s)

    doc = fitz.open(str(pdf))
    txt_parts: list[str] = []
    for i in range(min(pages, doc.page_count)):
        try:
            txt_parts.append(doc.load_page(i).get_text("text") or "")
        except Exception:
            continue

    txt = "\n".join(txt_parts).strip()
    if not txt:
        return PdfTextProbe(chars=0, letters=0, spaces=0, replacement=0, control=0, sample="")

    letters = sum(ch.isalpha() for ch in txt)
    spaces = sum(ch.isspace() for ch in txt)
    replacement = txt.count("\uFFFD")
    control = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", txt))
    sample = txt[:400].replace("\n", " ")
    return PdfTextProbe(
        chars=len(txt),
        letters=letters,
        spaces=spaces,
        replacement=replacement,
        control=control,
        sample=sample,
    )


def decide_ocr_mode(probe: PdfTextProbe) -> OcrMode:
    if probe.chars == 0 or probe.letters < 40:
        return "force_ocr"

    repl_ratio = probe.replacement / max(1, probe.chars)
    ctrl_ratio = probe.control / max(1, probe.chars)

    if probe.chars >= 800 and probe.letters >= 250 and repl_ratio <= 0.01 and ctrl_ratio <= 0.002:
        return "none"

    return "strip_existing_ocr"


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument("--paper_id", type=str, default=None)
    ap.add_argument("--input_dir", type=Path, default=Path("data/papers"))
    ap.add_argument("--output_root", type=Path, default=Path("data/extraction"))

    ap.add_argument("--ollama_base_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama_model", type=str, default="llama3.1")

    ap.add_argument("--normalize_rules", type=Path, default=Path("data/normalize_md.yaml"))

    ap.add_argument("--auto_ocr", action="store_true", default=True)
    ap.add_argument("--no_auto_ocr", dest="auto_ocr", action="store_false")
    ap.add_argument("--force_ocr", action="store_true")
    ap.add_argument("--strip_existing_ocr", action="store_true")

    ap.add_argument("--marker_llm", action="store_true")
    ap.add_argument("--redo_inline_math", action="store_true")
    ap.add_argument("--disable_image_extraction", action="store_true")
    ap.add_argument("--save_marker_json", action="store_true")

    ap.add_argument("--extract_images", action="store_true")
    ap.add_argument("--images_dir_name", type=str, default="marker_images")

    ap.add_argument("--max_pdfs", type=int, default=0)
    ap.add_argument("--continue_on_error", action="store_true")

    # ключевое: поэтапно
    ap.add_argument("--only_stage", choices=["all", "split", "marker", "stitch", "normalize"], default="all")

    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    ensure_ollama_up(args.ollama_base_url)

    # pdf list
    if args.pdf is not None:
        pdfs = [args.pdf]
    else:
        pdfs = sorted(args.input_dir.glob("*.pdf"))
        if args.max_pdfs and args.max_pdfs > 0:
            pdfs = pdfs[: args.max_pdfs]

    if not pdfs:
        print("No PDFs found.")
        return

    for pdf in pdfs:
        paper = args.paper_id if (args.pdf is not None and args.paper_id) else pdf.stem
        paper_dir = args.output_root / paper

        pages_dir = paper_dir / "pages"
        md_pages_dir = paper_dir / "md_pages"
        marker_work_dir = paper_dir / "marker_work"   # marker internaly worked with every page
        stitched_dir = paper_dir / "stitched"
        norm_dir = paper_dir / "normalized"
        images_dir = paper_dir / args.images_dir_name

        for d in [pages_dir, md_pages_dir, marker_work_dir, stitched_dir, norm_dir, images_dir]:
            d.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        print(f"\n=== {pdf.name} ({paper}) ===", flush=True)

        probe = probe_pdf_text(pdf)
        if args.force_ocr:
            ocr_mode: OcrMode = "force_ocr"
        elif args.strip_existing_ocr:
            ocr_mode = "strip_existing_ocr"
        elif args.auto_ocr:
            ocr_mode = decide_ocr_mode(probe)
        else:
            ocr_mode = "none"

        redo_inline_math_effective = bool(args.redo_inline_math and args.marker_llm)

        try:
            # -------- split
            if args.only_stage in ("all", "split"):
                run_py([
                    sys.executable, "-m", "perov.step1a_split_joint",
                    "split",
                    str(pdf),
                    "--out_dir", str(pages_dir),
                ])

            # -------- marker per page -> md_pages/page_XXXX.md
            if args.only_stage in ("all", "marker"):
                page_pdfs = list_page_pdfs(pages_dir)
                print(
                    f"[INFO] OCR mode={ocr_mode} marker_llm={args.marker_llm} redo_inline_math={redo_inline_math_effective}",
                    flush=True,
                )

                for page_pdf in page_pdfs:
                    # new folder for processed pages
                    work = marker_work_dir / page_pdf.stem
                    if work.exists():
                        # cleaning after update
                        for p in work.glob("*"):
                            pass
                    work.mkdir(parents=True, exist_ok=True)

                    cmd = [
                        sys.executable, "-m", "perov.step1b_marker",
                        "--pdf", str(page_pdf),
                        "--out_dir", str(work),
                        "--md_out_dir", str(md_pages_dir),
                    ]

                    if ocr_mode == "force_ocr":
                        cmd.append("--force_ocr")
                    elif ocr_mode == "strip_existing_ocr":
                        pass

                    if args.disable_image_extraction:
                        cmd.append("--disable_image_extraction")
                    if redo_inline_math_effective:
                        cmd.append("--redo_inline_math")

                    if args.marker_llm:
                        cmd += [
                            "--marker_llm",
                            "--ollama_base_url", args.ollama_base_url,
                            "--ollama_model", args.ollama_model,
                        ]

                    if args.save_marker_json:
                        cmd.append("--save_marker_json")

                    if args.extract_images:
                        cmd += ["--extract_images", "--images_dir", str(images_dir)]

                    run_py(cmd)

            # -------- stitch
            stitched_md = stitched_dir / "paper.raw.md"
            if args.only_stage in ("all", "stitch"):
                run_py([
                    sys.executable, "-m", "perov.step1a_split_joint",
                    "stitch",
                    str(md_pages_dir),
                    "--md_glob", "page_*.md",
                    "--out_md", str(stitched_md),
                ])

            # -------- normalize
            normalized_md = norm_dir / "paper.normalized.md"
            if args.only_stage in ("all", "normalize"):
                if not stitched_md.exists():
                    raise FileNotFoundError(f"Missing stitched md: {stitched_md}")
                run_py([
                    sys.executable, "-m", "perov.step1c_normalization",
                    str(stitched_md),
                    "--rules", str(args.normalize_rules),
                    "--out_md", str(normalized_md),
                ])

            meta = {
                "paper": paper,
                "source_pdf": str(pdf.resolve()),
                "created_at_utc": utc_now_iso(),
                "probe": probe.__dict__,
                "ocr_mode": ocr_mode,
                "dirs": {
                    "paper_dir": str(paper_dir),
                    "pages_dir": str(pages_dir),
                    "md_pages_dir": str(md_pages_dir),
                    "stitched_dir": str(stitched_dir),
                    "norm_dir": str(norm_dir),
                    "images_dir": str(images_dir),
                },
                "files": {
                    "stitched_md": str(stitched_md) if stitched_md.exists() else "",
                    "normalized_md": str(normalized_md) if normalized_md.exists() else "",
                },
                "flags": {
                    "marker_llm": args.marker_llm,
                    "redo_inline_math": args.redo_inline_math,
                    "extract_images": args.extract_images,
                    "disable_image_extraction": args.disable_image_extraction,
                },
            }
            (paper_dir / "meta_step1.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            dt = time.time() - t0
            print(f"[OK] step1 pipeline done in {dt/60:.1f} min", flush=True)

        except Exception as e:
            dt = time.time() - t0
            print(f"[ERROR] {pdf.name} failed after {dt/60:.1f} min: {e}", flush=True)
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
