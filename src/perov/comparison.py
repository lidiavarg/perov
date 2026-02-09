#!/usr/bin/env python3
import argparse
import base64
import json
import re
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Any


# ----------------------------
# Helpers
# ----------------------------
def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def require(cmd_name: str) -> None:
    if shutil.which(cmd_name) is None:
        raise SystemExit(
            f"[ERROR] '{cmd_name}' not found in PATH.\n"
            f"Install it (or run in the right venv) and try again."
        )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sniff_ext(b: bytes) -> str:
    if b.startswith(b"\x89PNG"):
        return "png"
    if b[:2] == b"\xff\xd8":
        return "jpg"
    if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
        return "gif"
    if b.startswith(b"%PDF"):
        return "pdf"
    return "bin"


# ----------------------------
# Docling: maximal image export (best-effort)
# ----------------------------
UNI_GLYPH_RE = re.compile(r"/uni[0-9A-Fa-f]{4}")

def _looks_like_base64(s: str) -> bool:
    # conservative: long and base64 charset
    if len(s) < 2000:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s) is not None


def _save_bytes(images_out: Path, stem: str, idx: int, data: bytes) -> Path:
    images_out.mkdir(parents=True, exist_ok=True)
    ext = sniff_ext(data)
    p = images_out / f"{stem}_{idx:04d}.{ext}"
    p.write_bytes(data)
    return p


def _try_export_via_document_methods(doc: Any, docling_out: Path, notes: list[str]) -> int:
    """
    Some docling versions expose export methods that can write assets.
    We try a few common patterns.
    Returns count of files created (best-effort estimate).
    """
    created = 0
    images_out = docling_out / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Candidate methods that might exist across versions:
    # - export(output_dir=...)
    # - export_to_dir(...)
    # - export_assets(...)
    # - export_figures(...)
    candidates = [
        ("export", {"output_dir": str(docling_out)}),
        ("export_to_dir", {"output_dir": str(docling_out)}),
        ("export_assets", {"output_dir": str(images_out)}),
        ("export_figures", {"output_dir": str(images_out)}),
        ("save_assets", {"output_dir": str(images_out)}),
    ]

    for fn_name, kwargs in candidates:
        fn = getattr(doc, fn_name, None)
        if not callable(fn):
            continue
        try:
            before = set(p.resolve() for p in images_out.glob("*"))
            fn(**kwargs)  # type: ignore
            after = set(p.resolve() for p in images_out.glob("*"))
            delta = len(after - before)
            created += delta
            notes.append(f"Docling: called document.{fn_name}({kwargs}); new files in images/: {delta}")
        except TypeError as e:
            notes.append(f"Docling: document.{fn_name} exists but signature mismatch: {e}")
        except Exception as e:
            notes.append(f"Docling: document.{fn_name} failed: {type(e).__name__}: {e}")

    return created


def _try_export_from_collections(doc: Any, images_out: Path, notes: list[str]) -> int:
    """
    Try doc.images / doc.figures / doc.assets collections.
    Save any raw bytes fields.
    """
    saved = 0
    images_out.mkdir(parents=True, exist_ok=True)

    for attr in ("images", "figures", "assets", "resources"):
        coll = getattr(doc, attr, None)
        if coll is None:
            continue

        try:
            # some are iterables, some are dict-like
            items = coll.values() if isinstance(coll, dict) else coll
            n_seen = 0
            for item in items:
                n_seen += 1

                # try common data fields
                for field in ("data", "bytes", "content", "blob", "raw", "payload"):
                    data = getattr(item, field, None)
                    if isinstance(data, (bytes, bytearray)):
                        saved += 1
                        _save_bytes(images_out, f"{attr}", saved, bytes(data))
                        break

                # sometimes dicts
                if isinstance(item, dict):
                    for k in ("data", "bytes", "content", "blob", "raw", "payload"):
                        v = item.get(k)
                        if isinstance(v, (bytes, bytearray)):
                            saved += 1
                            _save_bytes(images_out, f"{attr}", saved, bytes(v))
                            break
                        if isinstance(v, str) and _looks_like_base64(v):
                            try:
                                b = base64.b64decode(v, validate=False)
                                # only save if seems like an image
                                if sniff_ext(b) in ("png", "jpg", "gif"):
                                    saved += 1
                                    _save_bytes(images_out, f"{attr}", saved, b)
                            except Exception:
                                pass

            notes.append(f"Docling: inspected document.{attr} (items seen: {n_seen})")
        except Exception as e:
            notes.append(f"Docling: inspecting document.{attr} failed: {type(e).__name__}: {e}")

    return saved


def _try_export_from_serialized(doc: Any, images_out: Path, notes: list[str]) -> int:
    """
    Try doc.export()/to_dict()/as_dict() and crawl for embedded bytes/base64.
    """
    images_out.mkdir(parents=True, exist_ok=True)
    saved = 0
    doc_dict = None

    for fn_name in ("export", "to_dict", "as_dict", "export_to_dict", "model_dump"):
        fn = getattr(doc, fn_name, None)
        if not callable(fn):
            continue
        try:
            # model_dump from pydantic often supports mode="json"
            if fn_name == "model_dump":
                doc_dict = fn(mode="python")  # type: ignore
            else:
                doc_dict = fn()  # type: ignore
            notes.append(f"Docling: got serialized document via {fn_name}()")
            break
        except Exception as e:
            notes.append(f"Docling: {fn_name}() failed: {type(e).__name__}: {e}")

    if not isinstance(doc_dict, (dict, list)):
        notes.append("Docling: no serialized dict/list available for deep image crawl.")
        return 0

    def walk(x: Any):
        nonlocal saved
        if isinstance(x, dict):
            for _, v in x.items():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            if isinstance(x, (bytes, bytearray)):
                b = bytes(x)
                ext = sniff_ext(b)
                if ext in ("png", "jpg", "gif"):
                    saved += 1
                    _save_bytes(images_out, "serialized", saved, b)
            elif isinstance(x, str):
                if _looks_like_base64(x):
                    try:
                        b = base64.b64decode(x, validate=False)
                        ext = sniff_ext(b)
                        if ext in ("png", "jpg", "gif"):
                            saved += 1
                            _save_bytes(images_out, "serialized", saved, b)
                    except Exception:
                        pass

    walk(doc_dict)
    notes.append(f"Docling: deep crawl saved {saved} embedded image payload(s).")
    return saved


def extract_docling_md_and_images(pdf: Path, docling_out: Path) -> dict[str, Any]:
    """
    Convert PDF with docling, export markdown, and best-effort export images.
    Returns debug info (notes, method availability, counts).
    """
    notes: list[str] = []
    docling_out.mkdir(parents=True, exist_ok=True)

    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:
        raise SystemExit(f"[ERROR] Docling import failed in this environment: {type(e).__name__}: {e}")

    converter = DocumentConverter()

    # absolute path to avoid URL/path issues
    src = str(pdf.resolve())
    result = converter.convert(src)
    doc = result.document

    # ---- markdown export (API varies) ----
    md_text = None
    used_md_fn = None
    for fn_name in ("export_to_markdown", "to_markdown", "export_markdown", "as_markdown"):
        fn = getattr(doc, fn_name, None)
        if callable(fn):
            md_text = fn()
            used_md_fn = fn_name
            break

    if md_text is None:
        for fn_name in ("export_to_text", "to_text", "export_text", "as_text"):
            fn = getattr(doc, fn_name, None)
            if callable(fn):
                md_text = fn()
                used_md_fn = fn_name
                notes.append("Docling: markdown method not found; fell back to plain text export.")
                break

    if md_text is None:
        raise SystemExit("[ERROR] Docling document has no markdown/text export method in this version.")

    md_path = docling_out / f"{pdf.stem}.md"
    write_text(md_path, md_text)

    # ---- maximal image export ----
    images_out = docling_out / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Try higher-level exporter methods first
    created_by_methods = _try_export_via_document_methods(doc, docling_out, notes)

    # Then try reading collections (images/figures/assets)
    saved_from_collections = _try_export_from_collections(doc, images_out, notes)

    # Then try serialized crawl (export/to_dict)
    saved_from_serialized = _try_export_from_serialized(doc, images_out, notes)

    # Heuristic: detect docling glyph artifacts in MD (good to know)
    artifact_hits = len(UNI_GLYPH_RE.findall(md_text))

    # Gather debug snapshot: what attrs exist
    attrs = sorted(set(dir(doc)))
    interesting = {
        "image_like_attrs": [a for a in attrs if any(k in a.lower() for k in ("image", "figure", "asset", "resource"))][:200],
        "markdown_like_attrs": [a for a in attrs if "mark" in a.lower()][:200],
        "text_like_attrs": [a for a in attrs if "text" in a.lower()][:200],
        "export_like_attrs": [a for a in attrs if "export" in a.lower()][:200],
    }

    summary = {
        "md_path": str(md_path),
        "images_dir": str(images_out),
        "md_export_method": used_md_fn,
        "created_by_doc_methods": created_by_methods,
        "saved_from_collections": saved_from_collections,
        "saved_from_serialized": saved_from_serialized,
        "total_files_in_images_dir": len(list(images_out.glob("*"))),
        "md_uni_artifact_tokens": artifact_hits,
        "notes": notes,
        "doc_attrs": interesting,
    }
    return summary


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Marker then Docling for the same PDF, saving md+assets into separate folders."
    )
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--paper_id", required=True, help="Paper id (used for output folder names)")
    ap.add_argument("--out_root", default="data/extraction", help="Root output directory")

    # Marker options
    ap.add_argument("--marker_cmd", default="marker_single",
                    help="Marker CLI command (default: marker_single)")
    ap.add_argument("--marker_extra", default="",
                    help="Extra args passed to Marker (string). Example: '--ocr none --langs en'")

    args = ap.parse_args()

    pdf = Path(args.pdf).expanduser().resolve()
    if not pdf.exists():
        raise SystemExit(f"[ERROR] PDF not found: {pdf}")

    out_root = Path(args.out_root).expanduser().resolve()
    marker_out = out_root / args.paper_id / "marker"
    docling_out = out_root / args.paper_id / "docling"
    marker_out.mkdir(parents=True, exist_ok=True)
    docling_out.mkdir(parents=True, exist_ok=True)

    # Ensure marker exists
    require(args.marker_cmd)

    # ---- 1) Marker ----
    marker_cmd = [
        args.marker_cmd,
        str(pdf),
        "--output_format", "markdown",
        "--output_dir", str(marker_out),
    ]
    if args.marker_extra.strip():
        marker_cmd += args.marker_extra.split()

    print(f"[INFO] Marker output: {marker_out}")
    run(marker_cmd)

    # ---- 2) Docling (Python API + maximal image export) ----
    print(f"[INFO] Docling output: {docling_out}")
    docling_summary = extract_docling_md_and_images(pdf, docling_out)

    # Save debug info for reproducibility
    debug_path = docling_out / "docling_debug.json"
    write_json(debug_path, docling_summary)

    print("\n[OK] Done.")
    print(f" - Marker:      {marker_out}")
    print(f" - Docling:     {docling_out}")
    print(f" - Docling md:  {docling_summary.get('md_path')}")
    print(f" - Docling img: {docling_summary.get('images_dir')} (files: {docling_summary.get('total_files_in_images_dir')})")
    print(f" - Debug:       {debug_path}")


if __name__ == "__main__":
    main()
