#!/usr/bin/env python3
"""Extract Grok/Aurora images from the local parquet preference dataset.

The dataset stores two images per row with model names. We keep the image
whose model is ``aurora-20-1-25`` and save it into the hard-style fake corpus.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "datasets" / "xAI_Aurora_t2i_human_preferences" / "data"
OUT_DIR = ROOT / "datasets" / "prepared" / "hard_styles" / "fake"
META = ROOT / "datasets" / "prepared" / "hard_styles" / "manifest_grok.jsonl"
TARGET = 500
MODEL = "aurora-20-1-25"


def _guard_hashes() -> set[str]:
    guard: set[str] = set()
    fresh = ROOT / "datasets" / "prepared" / "fresh_test" / "manifest.jsonl"
    # fresh_test is in /tmp/opencode during this session, but keep this helper
    # harmless if the file is absent in the repo.
    for path in (ROOT / "datasets" / "prepared" / "swin_union" / "manifest.jsonl",
                 Path("/tmp/opencode/fresh_test/manifest.jsonl")):
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("sha256"):
                    guard.add(r["sha256"])
    return guard


def _load_existing() -> tuple[list[dict], set[str]]:
    records: list[dict] = []
    seen: set[str] = set()
    if META.exists():
        with META.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                records.append(r)
                if r.get("sha256"):
                    seen.add(r["sha256"])
    if OUT_DIR.exists():
        for f in OUT_DIR.iterdir():
            stem = f.stem.split("__")[-1]
            if len(stem) == 16:
                seen.add(stem)
    return records, seen


def main() -> int:
    guard = _guard_hashes()
    records, seen = _load_existing()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    got = sum(1 for r in records if r.get("generator") == "grok")
    if got >= TARGET:
        print(f"already have {got} grok images")
        return 0

    parquet_files = sorted(SRC_DIR.glob("train-*.parquet"))
    print(f"parquet files: {len(parquet_files)}")

    def guard_hit(dig: str) -> bool:
        if dig in seen or dig in guard:
            return True
        return any(dig[:16] == g[:16] for g in guard)

    for p in parquet_files:
        if got >= TARGET:
            break
        pf = pq.ParquetFile(p)
        cols = [c for c in ("model1", "model2", "image1", "image2") if c in pf.schema_arrow.names]
        if len(cols) < 4:
            continue
        print(f"reading {p.name} rows={pf.metadata.num_rows}")
        for rg in range(pf.num_row_groups):
            if got >= TARGET:
                break
            tbl = pf.read_row_group(rg, columns=cols)
            m1 = tbl.column("model1").to_pylist()
            m2 = tbl.column("model2").to_pylist()
            i1 = tbl.column("image1").to_pylist()
            i2 = tbl.column("image2").to_pylist()
            for idx in range(len(m1)):
                if got >= TARGET:
                    break
                if m1[idx] == MODEL:
                    blob = i1[idx]["bytes"]
                elif m2[idx] == MODEL:
                    blob = i2[idx]["bytes"]
                else:
                    continue
                if not blob:
                    continue
                dig = hashlib.sha256(blob).hexdigest()
                if guard_hit(dig):
                    continue
                try:
                    im = Image.open(io.BytesIO(blob)).convert("RGB")
                except Exception:
                    continue
                if max(im.size) < 400:
                    continue
                if min(im.size) > 2048:
                    im.thumbnail((2048, 2048))
                fp = OUT_DIR / f"grok__{dig[:16]}.jpg"
                im.save(fp, "JPEG", quality=92)
                records.append({
                    "path": str(fp),
                    "label": "fake",
                    "generator": "grok",
                    "origin": MODEL,
                    "sha256": dig,
                    "width": im.size[0],
                    "height": im.size[1],
                })
                seen.add(dig)
                got += 1
        print(f"  -> grok saved {got}")

    META.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"done grok={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
