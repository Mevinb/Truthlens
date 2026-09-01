#!/usr/bin/env python3
"""Validate the exact Swin training manifest before a long training run.

The default full check parses every row, verifies every image can be decoded,
recomputes stored-file SHA-256 values, and rejects label or split leakage. Use
``--quick`` only when a full validation was already completed for an unchanged
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT_DIR / "datasets" / "prepared" / "swin_union" / "manifest.jsonl"
DEFAULT_REPORT = ROOT_DIR / "results" / "metrics" / "swin_dataset_verification.json"
VALID_SPLITS = {"train", "val", "test", "holdout"}
VALID_LABELS = {"real", "fake"}
REQUIRED_FIELDS = {"path", "split", "label", "source", "generator", "sha256", "corpus"}


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _inspect_image(item: tuple[int, Path]) -> dict:
    index, path = item
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image.convert("RGB")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions {width}x{height}")
        return {"index": index, "sha256": digest.hexdigest(),
                "width": width, "height": height, "error": None}
    except Exception as exc:  # noqa: BLE001 - validation must report every bad file
        return {"index": index, "sha256": None, "width": None,
                "height": None, "error": f"{type(exc).__name__}: {exc}"}


def _limited(items: Iterable, limit: int = 20) -> list:
    result = []
    for item in items:
        result.append(item)
        if len(result) >= limit:
            break
    return result


def verify_manifest(manifest: Path, *, full: bool = True, workers: int = 8,
                    progress_every: int = 5000,
                    required_corpora: dict[str, int] | None = None) -> dict:
    """Return a JSON-serialisable verification report."""
    manifest = Path(manifest).resolve()
    started = time.time()
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    paths: list[Path] = []
    manifest_digest = hashlib.sha256()

    if not manifest.is_file():
        return {"ok": False, "manifest": str(manifest),
                "errors": [f"manifest does not exist: {manifest}"], "warnings": []}

    with manifest.open("rb") as fh:
        for line_number, raw_line in enumerate(fh, 1):
            manifest_digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            missing = sorted(REQUIRED_FIELDS - row.keys())
            if missing:
                errors.append(f"line {line_number}: missing fields {missing}")
            if row.get("split") not in VALID_SPLITS:
                errors.append(f"line {line_number}: invalid split {row.get('split')!r}")
            if row.get("label") not in VALID_LABELS:
                errors.append(f"line {line_number}: invalid label {row.get('label')!r}")
            path_value = row.get("path")
            if not isinstance(path_value, str) or not path_value:
                errors.append(f"line {line_number}: path must be a non-empty string")
                path = Path(f"/__invalid_path_{line_number}")
            else:
                path = _resolve_path(path_value)
            row["_line"] = line_number
            rows.append(row)
            paths.append(path)

    by_split = Counter(row.get("split", "?") for row in rows)
    by_label = Counter(row.get("label", "?") for row in rows)
    by_split_label = Counter(
        f"{row.get('split', '?')}/{row.get('label', '?')}" for row in rows)
    by_corpus = Counter(row.get("corpus", "?") for row in rows)
    for corpus, minimum in (required_corpora or {}).items():
        if by_corpus[corpus] < minimum:
            errors.append(
                f"required corpus {corpus!r} has {by_corpus[corpus]} rows; "
                f"need at least {minimum}")
    duplicate_paths = [str(path) for path, count in Counter(paths).items() if count > 1]
    if duplicate_paths:
        errors.append(f"duplicate manifest paths: {len(duplicate_paths)}; "
                      f"examples={duplicate_paths[:5]}")

    for split in ("train", "val"):
        for label in VALID_LABELS:
            if by_split_label[f"{split}/{label}"] == 0:
                errors.append(f"required class is empty: {split}/{label}")

    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        errors.append(f"missing image files: {len(missing_paths)}; "
                      f"examples={missing_paths[:5]}")

    decode_errors: list[dict] = []
    hash_mismatches: list[dict] = []
    duplicate_hash_groups: list[dict] = []
    if full and not missing_paths:
        print(f"[verify] decoding and hashing {len(paths):,} images with {workers} workers",
              flush=True)
        actual_hash_rows: dict[str, list[int]] = defaultdict(list)
        work = enumerate(paths)
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
            for completed, result in enumerate(pool.map(_inspect_image, work), 1):
                index = result["index"]
                row = rows[index]
                if result["error"]:
                    decode_errors.append({"line": row["_line"],
                                          "path": str(paths[index]),
                                          "error": result["error"]})
                else:
                    actual_hash = result["sha256"]
                    actual_hash_rows[actual_hash].append(index)
                    recorded = row.get("sha256")
                    if recorded != actual_hash:
                        hash_mismatches.append({
                            "line": row["_line"], "path": str(paths[index]),
                            "recorded": recorded, "actual": actual_hash,
                        })
                if progress_every and completed % progress_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(f"[verify] {completed:,}/{len(paths):,} images "
                          f"({completed / len(paths):.1%}) at "
                          f"{completed / elapsed:.1f} images/s", flush=True)

        for digest, indices in actual_hash_rows.items():
            if len(indices) < 2:
                continue
            duplicate_hash_groups.append({
                "sha256": digest,
                "splits": sorted({rows[i].get("split") for i in indices}),
                "labels": sorted({rows[i].get("label") for i in indices}),
                "paths": [str(paths[i]) for i in indices[:10]],
            })
        if decode_errors:
            errors.append(f"undecodable images: {len(decode_errors)}")
        if hash_mismatches:
            errors.append(f"stored-file sha256 mismatches: {len(hash_mismatches)}")
        if duplicate_hash_groups:
            errors.append(f"duplicate image-content groups: {len(duplicate_hash_groups)}")
    elif not full:
        warnings.append("quick mode skipped image decoding and content-hash verification")

    # The source distribution is intentionally diverse, but severe class skew
    # should be visible before training starts.
    for split in ("train", "val"):
        real = by_split_label[f"{split}/real"]
        fake = by_split_label[f"{split}/fake"]
        if real and fake and max(real, fake) / min(real, fake) > 2.0:
            warnings.append(f"{split} class ratio exceeds 2:1 (real={real}, fake={fake})")

    elapsed = round(time.time() - started, 2)
    return {
        "ok": not errors,
        "manifest": str(manifest),
        "manifest_sha256": manifest_digest.hexdigest(),
        "mode": "full" if full else "quick",
        "rows": len(rows),
        "counts": {
            "by_split": dict(sorted(by_split.items())),
            "by_label": dict(sorted(by_label.items())),
            "by_split_label": dict(sorted(by_split_label.items())),
            "by_corpus": dict(sorted(by_corpus.items())),
        },
        "duplicate_paths": len(duplicate_paths),
        "missing_paths": len(missing_paths),
        "decode_errors": _limited(decode_errors),
        "hash_mismatches": _limited(hash_mismatches),
        "duplicate_hash_groups": _limited(duplicate_hash_groups),
        "errors": errors,
        "warnings": warnings,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--quick", action="store_true",
                        help="check schema and file presence, but do not decode/hash images")
    parser.add_argument("--workers", type=int,
                        default=min(8, os.cpu_count() or 1))
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument(
        "--require-corpus", action="append", default=[], metavar="NAME=MIN_ROWS",
        help="require at least this many manifest rows from a corpus (repeatable)")
    args = parser.parse_args()

    required_corpora = {}
    try:
        for requirement in args.require_corpus:
            name, minimum = requirement.rsplit("=", 1)
            required_corpora[name] = int(minimum)
    except (ValueError, TypeError):
        parser.error("--require-corpus must use NAME=MIN_ROWS")

    report = verify_manifest(args.manifest, full=not args.quick,
                             workers=args.workers,
                             progress_every=args.progress_every,
                             required_corpora=required_corpora)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n[verify] dataset summary")
    print(json.dumps(report.get("counts", {}), indent=2))
    for warning in report.get("warnings", []):
        print(f"[verify] WARNING: {warning}")
    if report["ok"]:
        print(f"[verify] PASS: {report['rows']:,} rows in "
              f"{report['elapsed_seconds']:.2f}s")
        print(f"[verify] report: {args.report}")
        return 0
    for error in report.get("errors", []):
        print(f"[verify] ERROR: {error}", file=sys.stderr)
    print(f"[verify] FAIL; report: {args.report}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
