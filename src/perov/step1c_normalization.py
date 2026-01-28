#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1b_normalize_md.py

Goals:
- Stable markers:
    [[ASSET: ...]] => one per line; consecutive assets compact (no blank line between)
    [[PAGE:x:y]]   => hard chunk delimiter (blank lines around)
- Convert:
    <span id="page-x-y"></span> -> [[PAGE:x:y]]
    (#page-x-y)                 -> [[PAGE:x:y]]
- Normalize <sub>/<sup>:
    <sub>...</sub> -> _...
    <sup>...</sup> -> ^...   (but drops affiliation markers † ‡ *)
- Chemistry underscore artifacts removed in TEXT+MATH segments:
    CsPbBr_3 -> CsPbBr3
    MnO_2 -> MnO2
    Ti_3C_2T_x -> Ti3C2Tx
- YAML-driven regex rules with segmentation (text/math/code + links separately)
- Soft unwrap paragraphs (keeps structure lines intact) using unwrap_soft_linebreaks_v2
- Late HTML unescape + re-run sub/sup cleanup (handles double-escaped &lt;sup&gt;...)
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml

# ----------------------------
# Regex
# ----------------------------
PAGE_SPAN_RE = re.compile(r'<span\s+id="page-(\d+)-(\d+)"></span>', re.IGNORECASE)
PAGE_HASH_LINK_RE = re.compile(r"\(#page-(\d+)-(\d+)\)", re.IGNORECASE)

IMG_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')

SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE | re.DOTALL)
SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)

FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)

INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)

ASSET_LINE_START_FIX_RE = re.compile(r"(?m)(?<!\n)\[\[ASSET:")
ASSET_MARKER_END_RE = re.compile(r"(?m)(\[\[ASSET:[^\n]*\]\])(?!\n)")

PAGE_MARKER_END_RE = re.compile(r"(?m)(\[\[PAGE:\d+:\d+\]\])(?!\n)")
PAGE_BLOCK_RE = re.compile(r"\[\[PAGE:(\d+):(\d+)\]\]")

CHEM_US_DIGIT_RE = re.compile(r"\b([A-Za-z0-9]*[A-Z][A-Za-z0-9]*?)_([0-9]+)\b")
CHEM_US_X_RE = re.compile(r"\b([A-Za-z0-9]*[A-Z][A-Za-z0-9]*?)_x\b")
HAS_DIGIT_RE = re.compile(r"\d")
HAS_2CAPS_RE = re.compile(r"(?:[A-Z].*){2,}")

# ----------------------------
# YAML rules
# ----------------------------
FlagName = Literal["MULTILINE", "IGNORECASE", "DOTALL"]


def flags_from_names(names: List[FlagName]) -> int:
    f = 0
    for n in names:
        if n == "MULTILINE":
            f |= re.MULTILINE
        elif n == "IGNORECASE":
            f |= re.IGNORECASE
        elif n == "DOTALL":
            f |= re.DOTALL
    return f


@dataclass
class Rule:
    id: str
    stage: str
    scope: str  # all | text | math | code | links
    find: str
    replace: str
    flags: int


# ----------------------------
# Unwrap v2 helpers
# ----------------------------
HARD_BLOCK_RE = re.compile(
    r"^(?:"
    r"\s*$"
    r"|\s{0,3}```"
    r"|\s{0,3}\[\[ASSET:"
    r"|\s{0,3}\[\[PAGE:"
    r")"
)

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
TABLE_ROW_RE = re.compile(r"^\s{0,3}\|")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s")
HR_RE = re.compile(r"^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$")

UL_RE = re.compile(r"^\s{0,3}[-*+]\s+")
OL_RE = re.compile(r"^\s{0,3}\d+\.\s+")

CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"Figure|Fig\.|Table|Scheme|Chart|Equation|Eq\.|"
    r"Supporting Information|ASSOCIATED CONTENT|AUTHOR INFORMATION|"
    r"ACKNOWLEDGMENTS?|REFERENCES?|Notes?|Abstract|CONSPECTUS|"
    r"Materials and Methods|Materials & Methods|EXPERIMENTAL SECTION|"
    r"RESULTS AND DISCUSSION|Results and Discussion|CONCLUSIONS?|Conclusion"
    r")\b",
    re.IGNORECASE,
)

STANDALONE_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"\(\w+\)"
    r"|[a-zA-Z]\)"
    r"|[ivxIVX]+\)"
    r")\s*$"
)

SOFT_CONTINUATION_END_RE = re.compile(r".*[,;:/]\s*$")
STARTS_LOWER_OR_PUNCT_RE = re.compile(r"^[\s]*[a-z(“\"'\[]")
STARTS_DIGIT_RE = re.compile(r"^[\s]*\d")
STARTS_UPPER_RE = re.compile(r"^[\s]*[A-Z]")


def _is_hard_block(line: str) -> bool:
    return bool(HARD_BLOCK_RE.match(line))


def _is_structural_standalone(line: str) -> bool:
    return bool(
        HEADING_RE.match(line)
        or TABLE_ROW_RE.match(line)
        or BLOCKQUOTE_RE.match(line)
        or HR_RE.match(line)
        or CAPTION_RE.match(line)
    )


def _looks_like_list_item(line: str) -> bool:
    return bool(UL_RE.match(line) or OL_RE.match(line))


def _count_list_run(lines: List[str], i: int) -> int:
    n = 0
    j = i
    while j < len(lines):
        if lines[j].strip() == "":
            break
        if _looks_like_list_item(lines[j]):
            n += 1
            j += 1
            continue
        break
    return n


def _strip_list_prefix(line: str) -> str:
    if UL_RE.match(line):
        return UL_RE.sub("", line, count=1)
    if OL_RE.match(line):
        return OL_RE.sub("", line, count=1)
    return line


def _should_join(prev: str, nxt: str) -> bool:
    prev_s = prev.rstrip()
    nxt_s = nxt.strip()

    if not prev_s or not nxt_s:
        return False

    if _is_structural_standalone(nxt) or _is_structural_standalone(prev):
        return False

    # Hyphen dewrap: inter-\nface
    if prev_s.endswith("-") and re.match(r"^[A-Za-z]", nxt_s):
        return True

    # Unicode dash glue: 6243−\n6256 -> 6243–6256
    if prev_s and prev_s[-1] in ("−", "–", "—") and re.match(r"^[0-9A-Za-z]", nxt_s):
        return True

    if SOFT_CONTINUATION_END_RE.match(prev_s):
        return True

    if STARTS_LOWER_OR_PUNCT_RE.match(nxt_s) or STARTS_DIGIT_RE.match(nxt_s):
        return True

    if prev_s.endswith(".") and STARTS_UPPER_RE.match(nxt_s):
        return False

    if prev_s.endswith(("!", "?", ":")) and STARTS_UPPER_RE.match(nxt_s):
        return False

    if not prev_s.endswith((".", "!", "?", ":")):
        return True

    return False


def unwrap_soft_linebreaks_v2(md: str) -> str:
    lines = md.splitlines()
    out: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_hard_block(line):
            out.append(line)
            i += 1
            continue

        if _is_structural_standalone(line):
            out.append(line.rstrip())
            i += 1
            continue

        if line.strip() == "":
            out.append("")
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            continue

        # Preserve real list runs (>=2 items)
        if _looks_like_list_item(line):
            run = _count_list_run(lines, i)
            if run >= 2:
                for _ in range(run):
                    out.append(lines[i].rstrip())
                    i += 1
                continue

        buf = line.rstrip()
        if _looks_like_list_item(buf):
            buf = _strip_list_prefix(buf).strip()

        i += 1

        while i < len(lines):
            nxt = lines[i]

            if _is_hard_block(nxt) or nxt.strip() == "":
                break
            if _is_structural_standalone(nxt):
                break
            if _looks_like_list_item(nxt) and _count_list_run(lines, i) >= 2:
                break

            nxt_stripped = nxt.strip()

            # Keep standalone labels separate (often figure panel markers)
            if STANDALONE_LABEL_RE.match(nxt_stripped):
                break

            if _should_join(buf, nxt):
                if buf.endswith("-") and re.match(r"^[A-Za-z]", nxt_stripped):
                    buf = buf[:-1] + nxt_stripped
                elif buf and buf[-1] in ("−", "–", "—") and re.match(r"^[0-9A-Za-z]", nxt_stripped):
                    buf = buf[:-1].rstrip() + "–" + nxt_stripped.lstrip()
                else:
                    buf = buf + " " + nxt_stripped
                i += 1
                continue

            break

        out.append(buf)

        if i < len(lines) and lines[i].strip() == "":
            out.append("")
            while i < len(lines) and lines[i].strip() == "":
                i += 1

    return "\n".join(out) + "\n"


# ----------------------------
# YAML loader
# ----------------------------
def load_rules(yaml_path: Path) -> Tuple[List[str], List[Rule]]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict")

    defaults = cfg.get("defaults", {}) or {}
    default_flags = defaults.get("flags", ["MULTILINE"])
    if not isinstance(default_flags, list):
        default_flags = ["MULTILINE"]

    stages = cfg.get("stages", []) or []
    if not stages:
        stages = ["pre", "structure", "links", "assets", "math", "pages", "chemistry", "post"]

    rules_raw = cfg.get("rules", []) or []
    rules: List[Rule] = []
    for r in rules_raw:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", "")).strip()
        if not rid:
            continue
        stage = str(r.get("stage", "post")).strip()
        scope = str(r.get("scope", "all")).strip()
        find = str(r.get("find", ""))
        repl = str(r.get("replace", ""))

        fl = r.get("flags", default_flags)
        if not isinstance(fl, list):
            fl = default_flags
        flags = flags_from_names([str(x) for x in fl])

        rules.append(Rule(id=rid, stage=stage, scope=scope, find=find, replace=repl, flags=flags))

    stage_index = {s: i for i, s in enumerate(stages)}
    rules.sort(key=lambda x: (stage_index.get(x.stage, 10_000), x.id))
    return stages, rules


# ----------------------------
# Segmentation
# ----------------------------
SegmentType = Literal["text", "math", "code"]


@dataclass
class Segment:
    kind: SegmentType
    text: str


def split_code_blocks(md: str) -> List[Segment]:
    parts: List[Segment] = []
    last = 0
    for m in FENCED_CODE_RE.finditer(md):
        if m.start() > last:
            parts.append(Segment("text", md[last:m.start()]))
        parts.append(Segment("code", m.group(0)))
        last = m.end()
    if last < len(md):
        parts.append(Segment("text", md[last:]))
    return parts


def split_inline_math(text: str) -> List[Segment]:
    segs: List[Segment] = []
    last = 0
    for m in INLINE_MATH_RE.finditer(text):
        if m.start() > last:
            segs.append(Segment("text", text[last:m.start()]))
        segs.append(Segment("math", m.group(0)))
        last = m.end()
    if last < len(text):
        segs.append(Segment("text", text[last:]))
    return segs


def split_math(text: str) -> List[Segment]:
    segs: List[Segment] = []
    last = 0
    for m in BLOCK_MATH_RE.finditer(text):
        if m.start() > last:
            segs.extend(split_inline_math(text[last:m.start()]))
        segs.append(Segment("math", m.group(0)))
        last = m.end()
    if last < len(text):
        segs.extend(split_inline_math(text[last:]))
    return segs


def segment_md(md: str) -> List[Segment]:
    out: List[Segment] = []
    for seg in split_code_blocks(md):
        if seg.kind == "code":
            out.append(seg)
        else:
            out.extend(split_math(seg.text))
    return out


def apply_rules_to_segments(segs: List[Segment], rules: List[Rule]) -> None:
    compiled = [(r, re.compile(r.find, r.flags)) for r in rules]
    for rule, cre in compiled:
        for s in segs:
            if rule.scope == "all":
                s.text = cre.sub(rule.replace, s.text)
            elif rule.scope == "text" and s.kind == "text":
                s.text = cre.sub(rule.replace, s.text)
            elif rule.scope == "math" and s.kind == "math":
                s.text = cre.sub(rule.replace, s.text)
            elif rule.scope == "code" and s.kind == "code":
                s.text = cre.sub(rule.replace, s.text)
            # scope=links is applied before segmentation in normalize_md()


# ----------------------------
# Transforms
# ----------------------------
def html_sub_to_plain(m: re.Match) -> str:
    return f"_{m.group(1).strip()}"


def html_sup_to_plain(m: re.Match) -> str:
    v = m.group(1).strip()
    # Drop author affiliation markers (common in front matter)
    if v in {"†", "‡", "*"}:
        return ""
    return f"^{v}"


def normalize_page_anchors_to_block(text: str) -> str:
    def repl(m: re.Match) -> str:
        p, a = m.group(1), m.group(2)
        return f"\n\n[[PAGE:{p}:{a}]]\n\n"
    return PAGE_SPAN_RE.sub(repl, text)


def normalize_page_hash_links_to_block(text: str) -> str:
    def repl(m: re.Match) -> str:
        p, a = m.group(1), m.group(2)
        return f"\n\n[[PAGE:{p}:{a}]]\n\n"
    return PAGE_HASH_LINK_RE.sub(repl, text)


def extract_and_replace_images(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    assets: List[Dict[str, Any]] = []

    def repl(m: re.Match) -> str:
        path = m.group(1).strip()
        meta: Dict[str, Any] = {"raw_path": path}

        pm = re.search(
            r"_page_(\d+)_([A-Za-z]+_\d+)\.(png|jpg|jpeg|webp|tif|tiff)$",
            path,
            flags=re.IGNORECASE,
        )
        if pm:
            meta["page"] = int(pm.group(1))
            meta["id"] = pm.group(2)
            meta["ext"] = pm.group(3).lower()
            placeholder = f"[[ASSET: page={meta['page']}, id={meta['id']}, file={path}]]"
        else:
            placeholder = f"[[ASSET: file={path}]]"

        assets.append(meta)
        return f"\n{placeholder}\n"

    return IMG_RE.sub(repl, text), assets


def force_asset_lines(text: str) -> str:
    text = ASSET_LINE_START_FIX_RE.sub("\n[[ASSET:", text)
    text = ASSET_MARKER_END_RE.sub(r"\1\n", text)
    text = re.sub(r"\n{2,}\[\[ASSET:", "\n[[ASSET:", text)
    text = re.sub(r"\]\]\n\n\[\[ASSET:", "]]\n[[ASSET:", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def force_page_blocks(text: str) -> str:
    text = re.sub(r"(\S)\[\[PAGE:(\d+):(\d+)\]\]", r"\1\n\n[[PAGE:\2:\3]]", text)
    text = PAGE_BLOCK_RE.sub(lambda m: f"\n\n[[PAGE:{m.group(1)}:{m.group(2)}]]\n\n", text)
    text = PAGE_MARKER_END_RE.sub(r"\1\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_chem_underscores(s: str) -> str:
    """
    Remove underscore artifacts for chemistry-like tokens:
      TOKEN_3 -> TOKEN3
      TOKEN_x -> TOKENx
    Guarded to avoid touching common math like a_1.
    """
    def ok(tok: str) -> bool:
        return bool(HAS_DIGIT_RE.search(tok) or HAS_2CAPS_RE.search(tok))

    def repl_digit(m: re.Match) -> str:
        left, num = m.group(1), m.group(2)
        return f"{left}{num}" if ok(left) else m.group(0)

    def repl_x(m: re.Match) -> str:
        left = m.group(1)
        return f"{left}x" if ok(left) else m.group(0)

    s = CHEM_US_DIGIT_RE.sub(repl_digit, s)
    s = CHEM_US_X_RE.sub(repl_x, s)
    return s


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def normalize_md(md_text: str, yaml_rules_path: Optional[Path]) -> Tuple[str, List[Dict[str, Any]]]:
    t = md_text

    # 0) PAGE anchors -> markers
    t = normalize_page_anchors_to_block(t)
    t = normalize_page_hash_links_to_block(t)

    # 0.5) Early unescape HTML entities (do twice for double-escaped strings)
    t = html.unescape(t)
    t = html.unescape(t)

    # 1) html sub/sup -> plain markers
    t = SUB_RE.sub(html_sub_to_plain, t)
    t = SUP_RE.sub(html_sup_to_plain, t)

    # 2) images -> assets
    t, assets = extract_and_replace_images(t)
    t = force_asset_lines(t)

    # 3) YAML rules:
    if yaml_rules_path:
        _, rules = load_rules(yaml_rules_path)
        link_rules = [r for r in rules if r.scope == "links"]
        non_link_rules = [r for r in rules if r.scope != "links"]

        # Apply URL/links rules globally
        for r in link_rules:
            t = re.compile(r.find, r.flags).sub(r.replace, t)

        # Segment into text/math/code and apply scoped rules
        segs = segment_md(t)

        # Chemistry underscore normalization in TEXT + MATH segments
        for s in segs:
            if s.kind in ("text", "math"):
                s.text = normalize_chem_underscores(s.text)

        apply_rules_to_segments(segs, non_link_rules)
        t = "".join(s.text for s in segs)

    # 4) Enforce PAGE blocks after all transforms
    t = force_page_blocks(t)

    # 4.5) Late HTML unescape (some pipelines double-escape &lt;sup&gt;...)
    t = html.unescape(t)
    t = html.unescape(t)

    # Re-run sub/sup cleanup after late unescape
    t = SUB_RE.sub(html_sub_to_plain, t)
    t = SUP_RE.sub(html_sup_to_plain, t)

    # Also drop any leftover escaped sup markers (safety net)
    t = re.sub(r"(?i)\^?&lt;sup&gt;\s*[†‡*]\s*&lt;/sup&gt;", "", t)

    # 5) Robust unwrap (v2)
    t = unwrap_soft_linebreaks_v2(t)

    # 6) Final whitespace normalization
    t = normalize_whitespace(t)
    return t, assets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("md_file", type=str, help="Input markdown file")
    ap.add_argument("--rules", type=str, default=None, help="YAML rules file (normalize_md.yaml)")
    ap.add_argument("--out_md", type=str, default=None, help="Output normalized md")
    ap.add_argument("--out_assets", type=str, default=None, help="Output assets json")
    args = ap.parse_args()

    in_path = Path(args.md_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input md not found: {in_path}")

    rules_path = Path(args.rules) if args.rules else None
    if rules_path and not rules_path.exists():
        raise FileNotFoundError(f"Rules YAML not found: {rules_path}")

    raw = in_path.read_text(encoding="utf-8", errors="ignore")
    norm, assets = normalize_md(raw, rules_path)

    out_md = Path(args.out_md) if args.out_md else in_path.with_suffix(".normalized.md")
    out_md.write_text(norm, encoding="utf-8")

    if args.out_assets or assets:
        out_assets = Path(args.out_assets) if args.out_assets else in_path.with_suffix(".assets.json")
        out_assets.write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote: {out_md}")
    print(f"[OK] assets: {len(assets)}")
    if rules_path:
        print(f"[OK] rules: {rules_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())