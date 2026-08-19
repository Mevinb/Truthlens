#!/usr/bin/env python3
"""
TruthLens — Web Test Set Builder (non-Wikimedia)
=================================================
Downloads real photographs and AI-generated images from the open web into
``datasets/evaluation/internet/``, alongside whatever ``build_internet_testset.py`` managed
to collect from Wikimedia Commons. Same output directory and same manifest
schema, so ``diag_eval_all.py`` picks both up as one ``internet`` set and groups
results by the ``tag`` field.

Why this exists next to the Commons builder
-------------------------------------------
``upload.wikimedia.org`` rate-limits this IP hard: four attempts at an 8-second
request interval yielded four images before tripping twelve consecutive HTTP
429s. The Commons *API* answers fine — it is only the file host that refuses —
so the label-quality advantage of Commons (uploaders must declare AI generation)
is unreachable in bulk. This builder uses hosts that serve bulk traffic
willingly. Keep the Commons retry running in the background; if it ever clears,
its rows append to the same manifest and the eval board absorbs them.

Sources, and what each is worth
-------------------------------
``web-unsplash``   real. Unsplash originals, 2-12MP, straight off
                   ``images.unsplash.com``. Independent of every real source in
                   the training corpus — Megalith is Flickr CC0, OpenImages is
                   Google's crawl, ``bm-subnet-real-fullsize`` is an aggregated
                   feed. This is the honest false-alarm measurement: arbitrary
                   high-resolution web photographs, the exact thing a user
                   uploads and expects to be told is real.
``web-openverse``  real. Openverse's CC index, restricted to non-Wikimedia
                   providers (the Wikimedia-hosted rows point back at the
                   throttled host). Adds provider diversity behind Unsplash's
                   single house style, which matters because "looks like stock
                   photography" is itself a learnable shortcut.
``web-civitai``    fake, **mixed style — read this group separately**. Every
                   image on Civitai is generated, so the label is reliable, but
                   the API stopped returning prompt and ``baseModel`` metadata,
                   so there is no way to filter to photorealistic output. The
                   group therefore contains illustration and anime alongside
                   photoreal work. An anime image classified FAKE is an easy
                   win, so this group's accuracy is an *upper* bound on fake
                   recall and must not be read as photoreal performance. It
                   still earns its place: a stylised AI image called REAL is a
                   genuine failure, and this is the only independent
                   in-the-wild fake source available without an API key.

For photoreal fake performance, read ``holdout`` on the eval board instead —
``chatgpt4o_holdout`` is 1,202 original-byte ChatGPT PNGs from an unrelated
uploader, and ``liars_dividend`` is GPT-Image-2 output caption-paired with real
COCO photographs.

Normalisation: why every image is re-encoded here
-------------------------------------------------
Each class arrives from its own CDN, so storing original bytes would hand the
model the encoder as a class label — the precise confound the corpus audit was
built to catch, reintroduced in the one set meant to be immune to it. So every
image, both classes, is decoded and re-encoded through a single Pillow JPEG
encoder at a long edge and quality drawn from **distributions shared by both
classes**, seeded from the content hash. Container format, resolution and
quality then carry zero class information by construction, and the set spans
several realistic regimes instead of one artificial operating point.

The cost is that a fingerprint visible only in pristine bytes is invisible here.
That is the right trade for this set: almost every image reaching the app has
been through some platform's encoder, so this is the production regime. The
pristine regime is covered by ``holdout``, which stores original bytes.

Nothing here is training data. This set exists to be scored against; folding it
into training would destroy the only independent measurement available.

Usage
-----
    python dataset/build_web_testset.py --per-class 150
    python dataset/build_web_testset.py --report
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "evaluation" / "internet"
MANIFEST_PATH = OUTPUT_DIR / "manifest.jsonl"

UA = "TruthLens-research/1.0 (AI-image-detection evaluation; non-commercial)"

# Both classes are re-encoded from these two sets, chosen by content hash. The
# distributions are shared, so neither long edge nor quality can be read as a
# label. Four values each rather than a continuous range keeps the manifest
# legible and the audit panels readable.
LONG_EDGES = (768, 1024, 1280, 1600)
QUALITIES = (72, 80, 88, 94)

# Orientation mix enforced on **both** classes.
#
# Not a nicety. Unsplash Lite is overwhelmingly 3:2 landscape and Civitai is
# overwhelmingly portrait, so an unfiltered take of 150 per class hands over an
# orientation cue that separates the two almost perfectly — in the one set whose
# entire purpose is to be free of such cues. A first 12-per-class trial came back
# 12/12 landscape real against 11/20 portrait fake, which would have made any
# accuracy figure from this set meaningless.
#
# Aspect ratio survives the resize, so a candidate can be filtered on its
# original dimensions before it is downloaded — the API metadata is enough, and
# the rejected candidates cost nothing.
ORIENTATION_MIX = {"landscape": 0.40, "portrait": 0.40, "square": 0.20}


def orientation_of(w: int, h: int) -> str:
    if not w or not h:
        return "?"
    ratio = w / h
    return ("square" if 0.95 <= ratio <= 1.05
            else "landscape" if ratio > 1 else "portrait")


# Smallest original accepted. Below roughly 600px short side there is no native
# scale left for the detector's native-crop views to sample, so such an image
# measures the whole-frame path only and dilutes the set.
MIN_SHORT = 600

REQUEST_INTERVAL = 0.35        # polite, and far below what these CDNs allow
_last_request = [0.0]


def _throttle() -> None:
    wait = REQUEST_INTERVAL - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _fetch(url: str, retries: int = 3, timeout: int = 45) -> Optional[bytes]:
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < retries - 1:
                time.sleep(4.0 * (attempt + 1))
                continue
            if attempt == retries - 1:
                print(f"    ! HTTP {exc.code} {url[-52:]}", file=sys.stderr)
        except Exception as exc:                                   # noqa: BLE001
            if attempt == retries - 1:
                print(f"    ! {type(exc).__name__} {url[-52:]}", file=sys.stderr)
            else:
                time.sleep(1.5 * (attempt + 1))
    return None


def _api(url: str) -> dict:
    raw = _fetch(url, timeout=30)
    if raw is None:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:                                              # noqa: BLE001
        return {}


# ─── Candidate discovery ─────────────────────────────────────────────────────
def iter_unsplash(want: int) -> Iterator[dict]:
    """Unsplash originals, from the Unsplash Lite metadata dump on the Hub.

    The dump carries ``image_url`` plus native dimensions, and the bare URL (no
    query parameters) serves the original file — so the resize is done here,
    under one encoder, rather than by Unsplash's CDN under theirs.
    """
    from datasets import load_dataset

    ds = load_dataset("1aurent/unsplash-lite", split="train", streaming=True)
    seen = 0
    for row in ds:
        photo = row.get("photo") or {}
        w, h = photo.get("width") or 0, photo.get("height") or 0
        url = photo.get("image_url")
        if not url or min(w, h) < MIN_SHORT:
            continue
        yield {"url": url, "title": photo.get("id", ""), "tag": "web-unsplash",
               "label": "real", "source": "unsplash",
               "orig_width": w, "orig_height": h}
        seen += 1
        if seen >= want:
            return


# Subjects spread across the same ground the AI sets cover, so a score here
# cannot come from recognising subject matter.
OPENVERSE_QUERIES = ("portrait photograph", "landscape photograph",
                     "food photograph", "bird photograph",
                     "architecture photograph", "flower photograph",
                     "street photography", "mountain photograph")


def iter_openverse(want: int) -> Iterator[dict]:
    """Openverse CC index, excluding Wikimedia-hosted rows.

    Wikimedia rows resolve to ``upload.wikimedia.org``, which is the host that
    refuses bulk traffic — including them would spend the whole budget on 429s.
    """
    per_query = max(1, want // len(OPENVERSE_QUERIES)) + 2
    got = 0
    for query in OPENVERSE_QUERIES:
        taken = 0
        for page in range(1, 7):
            if taken >= per_query or got >= want:
                break
            data = _api("https://api.openverse.org/v1/images/?"
                        + urllib.parse.urlencode({
                            "q": query, "page_size": 20, "page": page,
                            "license_type": "all-cc", "mature": "false"}))
            results = data.get("results") or []
            if not results:
                break
            for r in results:
                if taken >= per_query or got >= want:
                    break
                if (r.get("provider") or "") == "wikimedia":
                    continue
                w, h = r.get("width") or 0, r.get("height") or 0
                url = r.get("url")
                if not url or min(w, h) < MIN_SHORT:
                    continue
                yield {"url": url, "title": (r.get("title") or "")[:120],
                       "tag": "web-openverse", "label": "real",
                       "source": f"openverse/{r.get('provider')}",
                       "orig_width": w, "orig_height": h}
                taken += 1
                got += 1


def iter_civitai(want: int) -> Iterator[dict]:
    """Civitai's public image feed. Every row is generated, so ``fake`` is safe.

    Paged by cursor. ``sort`` is varied across passes so the take is not one
    slice of one leaderboard, which would mean one week of one community's
    favourite checkpoint.
    """
    got = 0
    for sort in ("Most Reactions", "Newest", "Most Comments"):
        url = ("https://civitai.com/api/v1/images?"
               + urllib.parse.urlencode({"limit": 100, "nsfw": "None",
                                         "sort": sort}))
        for _ in range(12):
            if got >= want:
                return
            data = _api(url)
            items = data.get("items") or []
            if not items:
                break
            for it in items:
                if got >= want:
                    return
                w, h = it.get("width") or 0, it.get("height") or 0
                src = it.get("url")
                if not src or min(w, h) < MIN_SHORT or it.get("type") != "image":
                    continue
                yield {"url": src, "title": str(it.get("id", "")),
                       "tag": "web-civitai", "label": "fake", "source": "civitai",
                       "orig_width": w, "orig_height": h}
                got += 1
            nxt = (data.get("metadata") or {}).get("nextPage")
            if not nxt:
                break
            url = nxt


FETCHERS = {"unsplash": iter_unsplash, "openverse": iter_openverse,
            "civitai": iter_civitai}


# ─── Normalisation ───────────────────────────────────────────────────────────
def normalise(raw: bytes) -> Optional[Tuple[bytes, dict]]:
    """Re-encode through one JPEG encoder at a hash-chosen size and quality.

    The choice is seeded from the *content* hash, so it is reproducible, is
    identical for both classes' distributions, and cannot correlate with the
    label. Downscale only: upscaling a small image would invent no detail while
    destroying what little native texture it had.
    """
    from PIL import Image

    try:
        with Image.open(BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SHORT:
                return None
            rng = random.Random(hashlib.sha256(raw).hexdigest())
            long_edge = rng.choice(LONG_EDGES)
            quality = rng.choice(QUALITIES)
            if max(w, h) > long_edge:
                scale = long_edge / max(w, h)
                im = im.resize((max(1, round(w * scale)),
                                max(1, round(h * scale))), Image.LANCZOS)
            out = BytesIO()
            im.save(out, format="JPEG", quality=quality, subsampling=0)
            nw, nh = im.size
    except Exception:                                              # noqa: BLE001
        return None
    return out.getvalue(), {
        "width": nw, "height": nh, "stored_format": "JPEG",
        "encoded_long_edge": long_edge, "encoded_quality": quality,
        "orientation": ("square" if 0.95 <= nw / nh <= 1.05
                        else "landscape" if nw > nh else "portrait"),
    }


def load_seen() -> set:
    if not MANIFEST_PATH.exists():
        return set()
    out = set()
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.add(row["sha256"])
            if row.get("source_sha256"):
                out.add(row["source_sha256"])
    return out


# ─── Build ───────────────────────────────────────────────────────────────────
PLAN: Dict[str, List[Tuple[str, float]]] = {
    # source key → share of that class's budget
    "real": [("unsplash", 0.65), ("openverse", 0.35)],
    "fake": [("civitai", 1.0)],
}


def build(per_class: int) -> None:
    from tqdm import tqdm

    seen = load_seen()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = MANIFEST_PATH.open("a", encoding="utf-8")

    print("=" * 74)
    print("  Building web test set (non-Wikimedia)")
    print(f"  → {OUTPUT_DIR}")
    print(f"  target {per_class}/class · re-encoded JPEG, long edge "
          f"{LONG_EDGES}, q{QUALITIES}")
    print(f"  orientation mix per class: "
          f"{', '.join(f'{k} {v:.0%}' for k, v in ORIENTATION_MIX.items())}")
    print("=" * 74)

    # Orientation caps are per class and shared across that class's sources, so
    # both classes end with the same shape distribution regardless of which
    # provider happened to supply the images. Seeded from what is already on
    # disk, so a resumed or topped-up run corrects an existing skew instead of
    # adding to it.
    caps = {o: max(1, round(per_class * frac))
            for o, frac in ORIENTATION_MIX.items()}
    filled: Dict[Tuple[str, str], int] = collections.Counter()
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                filled[(row["label"], row.get("orientation", "?"))] += 1
        if filled:
            print("  already on disk: "
                  + "; ".join(f"{lab}/{shape} {n}/{caps.get(shape, 0)}"
                              for (lab, shape), n in sorted(filled.items())))

    try:
        for label, plan in PLAN.items():
            for key, share in plan:
                quota = max(1, int(round(per_class * share)))
                bar = tqdm(total=quota, desc=f"  {label}/{key:<12}", ncols=74,
                           unit="img", leave=True)
                taken = 0
                skipped_shape = 0
                # Over-request candidates generously: most are rejected on
                # orientation or the resolution floor. Discovery is one API page
                # or a parquet stream; only the accepted ones are downloaded.
                for cand in FETCHERS[key](quota * 12):
                    if taken >= quota:
                        break
                    shape = orientation_of(cand["orig_width"],
                                           cand["orig_height"])
                    if filled[(label, shape)] >= caps.get(shape, 0):
                        skipped_shape += 1
                        continue
                    raw = _fetch(cand["url"])
                    if not raw:
                        continue
                    src_digest = hashlib.sha256(raw).hexdigest()
                    if src_digest in seen:
                        continue
                    done = normalise(raw)
                    if done is None:
                        continue
                    payload, meta = done
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest in seen:
                        continue

                    out_dir = OUTPUT_DIR / label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    name = f"{cand['tag']}__{digest[:16]}.jpg"
                    (out_dir / name).write_bytes(payload)
                    seen.update((digest, src_digest))
                    filled[(label, meta["orientation"])] += 1
                    fh.write(json.dumps({
                        "sha256": digest,
                        "source_sha256": src_digest,
                        "path": f"{label}/{name}",
                        "label": label,
                        "source": cand["source"],
                        "tag": cand["tag"],
                        "title": cand["title"],
                        "url": cand["url"],
                        "orig_width": cand["orig_width"],
                        "orig_height": cand["orig_height"],
                        **meta,
                    }, sort_keys=True) + "\n")
                    fh.flush()
                    taken += 1
                    bar.update(1)
                bar.close()
                if taken < quota:
                    print(f"    {key}: {taken}/{quota} (candidates exhausted; "
                          f"{skipped_shape} skipped on orientation quota)")
    finally:
        fh.close()

    report()


def normalise_existing() -> None:
    """Re-encode any already-collected row that skipped the normaliser.

    The Commons builder stores original bytes, so its rows are a mix of PNG and
    JPEG at native resolution while everything from this builder is uniform
    JPEG. Left alone that is a format leak inside the set that exists to have
    none: PNG appears only in the fake class. This pass pushes those rows through
    the same encoder and rewrites their manifest entries in place.
    """
    if not MANIFEST_PATH.exists():
        print("Nothing to normalise.")
        return
    rows = [json.loads(l) for l in
            MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [r for r in rows if "encoded_long_edge" not in r]
    if not todo:
        print("Every row is already normalised.")
        return

    print(f"  normalising {len(todo)} row(s) collected with original bytes")
    out_rows: List[dict] = []
    dropped = 0
    for row in rows:
        if "encoded_long_edge" in row:
            out_rows.append(row)
            continue
        path = OUTPUT_DIR / row["path"]
        if not path.exists():
            dropped += 1
            continue
        done = normalise(path.read_bytes())
        if done is None:
            # Below the resolution floor, or undecodable. Dropping is right: an
            # un-normalised row would reintroduce the format leak.
            path.unlink(missing_ok=True)
            dropped += 1
            print(f"    dropped {row['path']} (below {MIN_SHORT}px or unreadable)")
            continue
        payload, meta = done
        digest = hashlib.sha256(payload).hexdigest()
        new_name = f"{row['tag']}__{digest[:16]}.jpg"
        (OUTPUT_DIR / row["label"] / new_name).write_bytes(payload)
        if new_name != path.name:
            path.unlink(missing_ok=True)
        row = {**row, **meta, "sha256": digest,
               "source_sha256": row.get("source_sha256", row["sha256"]),
               "path": f"{row['label']}/{new_name}"}
        out_rows.append(row)

    MANIFEST_PATH.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in out_rows),
        encoding="utf-8")
    print(f"  normalised {len(todo) - dropped}, dropped {dropped}")
    report()


def report() -> None:
    if not MANIFEST_PATH.exists():
        print("No test set yet — run without --report first.")
        return
    rows = [json.loads(l) for l in
            MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("Manifest is empty.")
        return

    print("\n" + "=" * 74)
    print(f"  INTERNET / WEB TEST SET — {len(rows)} images")
    print("=" * 74)
    for (label, tag), n in sorted(collections.Counter(
            (r["label"], r["tag"]) for r in rows).items()):
        print(f"  {label:<6}{tag:<24}{n:>5}")
    print("-" * 74)
    for label in ("real", "fake"):
        print(f"  {label:<6}{'TOTAL':<24}"
              f"{sum(1 for r in rows if r['label'] == label):>5}")

    # These columns must match between classes, or this set has the same leak it
    # was built to be immune to.
    print("\n  shortcut audit (columns should match between classes)")
    for name, keyfn in (("format", lambda r: r.get("stored_format", "?")),
                        ("orientation", lambda r: r.get("orientation", "?")),
                        ("long edge", lambda r: str(r.get("encoded_long_edge",
                                                          "native"))),
                        ("quality", lambda r: str(r.get("encoded_quality",
                                                        "native")))):
        for label in ("real", "fake"):
            sub = collections.Counter(keyfn(r) for r in rows
                                      if r["label"] == label)
            if sub:
                spread = "  ".join(f"{k}:{v}" for k, v in sorted(sub.items()))
                print(f"    {name:<12}{label:<6}{spread}")
    for label in ("real", "fake"):
        sub = [r for r in rows if r["label"] == label]
        if sub:
            mp = sum(r["width"] * r["height"] for r in sub) / len(sub) / 1e6
            print(f"    mean pixels {label:<6}{mp:>8.2f} MP")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--normalise-existing", action="store_true",
                    help="re-encode rows collected with original bytes (the "
                         "Commons builder's) so the whole set shares one encoder")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
        return 0
    if args.normalise_existing:
        normalise_existing()
        return 0
    build(args.per_class)
    return 0


if __name__ == "__main__":
    sys.exit(main())
