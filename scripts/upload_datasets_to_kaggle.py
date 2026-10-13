#!/usr/bin/env python3
"""
TruthLens — Kaggle Dataset Uploader
====================================
Uploads TruthLens datasets to Kaggle using `kagglehub.dataset_upload`.

Features:
  - Modular dataset catalog (modern_v2, multires, cifake, evaluation, raw sources, etc.)
  - Predefined bundles: core, prepared, raw, all
  - Strict exclusion filters to prevent uploading caches (_cache, .cache), pycache, temp files, or secrets
  - Pre-flight disk space verification before temporary zip archiving
  - Dry-run validation mode (--dry-run)
  - Custom directory upload support (--dir <path> --slug <slug>)
  - Automated credentials check from ~/.kaggle/kaggle.json, access_token, or env vars

Usage Examples:
  # 1. List all available datasets and sizes:
  python scripts/upload_datasets_to_kaggle.py --list

  # 2. Dry-run test on core training datasets:
  python scripts/upload_datasets_to_kaggle.py --group core --dry-run

  # 3. Upload a single dataset (e.g. evaluation or modern_v2):
  python scripts/upload_datasets_to_kaggle.py --dataset evaluation
  python scripts/upload_datasets_to_kaggle.py --dataset modern_v2

  # 4. Upload the core training & eval bundle:
  python scripts/upload_datasets_to_kaggle.py --group core

  # 5. Upload all modular datasets:
  python scripts/upload_datasets_to_kaggle.py --group all

  # 6. Upload a custom local folder:
  python scripts/upload_datasets_to_kaggle.py --dir path/to/dir --slug my-custom-dataset
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# ── Colors for CLI output ──────────────────────────────────────────────────
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Safe ignore patterns (applied in addition to kagglehub defaults)
SAFE_IGNORE_PATTERNS = [
    # Caches and virtual environments
    "_cache/",
    "_cache/*",
    "*/_cache/*",
    ".cache/",
    ".cache/*",
    "*/.cache/*",
    "__pycache__/",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    # Git
    ".git/",
    ".git/*",
    "*/.git/*",
    ".gitattributes",
    "*/.gitattributes",
    # Temporary and backup files
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.swp",
    "*.swo",
    "*~",
    # OS artifacts
    ".DS_Store",
    "*/.DS_Store",
    "Thumbs.db",
    "ehthumbs.db",
    # Sensitive credentials / keys
    "kaggle.json",
    "*/kaggle.json",
    "access_token",
    "*/access_token",
    ".env*",
    "*/.env*",
    "*.pem",
    "*.key",
]


@dataclass
class DatasetSpec:
    key: str
    path: Path
    slug: str
    title: str
    description: str
    groups: List[str] = field(default_factory=list)
    custom_ignore: List[str] = field(default_factory=list)


def build_catalog() -> dict[str, DatasetSpec]:
    """Builds the canonical TruthLens dataset catalog."""
    datasets_dir = PROJECT_ROOT / "datasets"

    catalog = {
        # Core prepared datasets
        "modern_v2": DatasetSpec(
            key="modern_v2",
            path=datasets_dir / "prepared" / "modern_v2",
            slug="truthlens-modern-v2",
            title="TruthLens Modern Multi-Generator Detector Corpus (v2)",
            description="High-resolution synthetic & real image corpus across Midjourney v6, SDXL, DALL-E 3, Flux.1, ChatGPT-4o.",
            groups=["core", "prepared", "all"],
        ),
        "multires": DatasetSpec(
            key="multires",
            path=datasets_dir / "prepared" / "multires",
            slug="truthlens-multires",
            title="TruthLens Multi-Resolution Scale-Robust Corpus",
            description="Multi-scale real and synthetic image pairs at native, 512px, and 256px resolutions.",
            groups=["core", "prepared", "all"],
        ),
        "hard_styles": DatasetSpec(
            key="hard_styles",
            path=datasets_dir / "prepared" / "hard_styles",
            slug="truthlens-hard-styles",
            title="TruthLens Hard Artistic Styles & Edge Cases",
            description="Challenging artistic styles, photorealistic renders, and ambiguous boundary cases.",
            groups=["core", "prepared", "all"],
        ),
        "swin_union": DatasetSpec(
            key="swin_union",
            path=datasets_dir / "prepared" / "swin_union",
            slug="truthlens-swin-union",
            title="TruthLens Swin-V2 Union Manifest",
            description="Unified multi-corpus manifest linking real and generated samples for Swin-V2 detector training.",
            groups=["core", "prepared", "all"],
        ),
        "evaluation": DatasetSpec(
            key="evaluation",
            path=datasets_dir / "evaluation",
            slug="truthlens-evaluation",
            title="TruthLens Independent Evaluation Benchmark",
            description="Out-of-distribution internet benchmark including Civitai and Unsplash image sets.",
            groups=["core", "all"],
        ),
        # Additional prepared corpora
        "modern_corpus": DatasetSpec(
            key="modern_corpus",
            path=datasets_dir / "prepared" / "modern_corpus",
            slug="truthlens-modern-corpus",
            title="TruthLens Modern Generator Corpus (v1)",
            description="Initial modern generator corpus collection.",
            groups=["prepared", "all"],
        ),
        "cifake": DatasetSpec(
            key="cifake",
            path=datasets_dir / "prepared" / "cifake",
            slug="truthlens-cifake",
            title="TruthLens CIFAKE Benchmark",
            description="Standard CIFAKE real and synthetic CIFAR-scale benchmark.",
            groups=["prepared", "all"],
        ),
        "highres_corpus": DatasetSpec(
            key="highres_corpus",
            path=datasets_dir / "prepared" / "highres_corpus",
            slug="truthlens-highres-corpus",
            title="TruthLens High-Resolution Corpus",
            description="Uncompressed high-resolution real and synthetic imagery.",
            groups=["prepared", "all"],
        ),
        "corpus": DatasetSpec(
            key="corpus",
            path=datasets_dir / "prepared" / "corpus",
            slug="truthlens-corpus",
            title="TruthLens Baseline Corpus",
            description="Baseline image corpus used for initial model verification.",
            groups=["prepared", "all"],
        ),
        "prepared_legacy": DatasetSpec(
            key="prepared_legacy",
            path=datasets_dir / "prepared" / "legacy_archive",
            slug="truthlens-prepared-legacy-archive",
            title="TruthLens Flattened Legacy Archive",
            description="Flattened, trainable representation of legacy real and synthetic archives.",
            groups=["prepared", "all"],
        ),
        "architectures": DatasetSpec(
            key="architectures",
            path=datasets_dir / "prepared" / "architectures",
            slug="truthlens-architectures",
            title="TruthLens Architecture Benchmark Manifests",
            description="Architecture benchmark splits and manifest metadata.",
            groups=["prepared", "all"],
        ),
        # External benchmark & preferences
        "xai_aurora": DatasetSpec(
            key="xai_aurora",
            path=datasets_dir / "xAI_Aurora_t2i_human_preferences",
            slug="truthlens-xai-aurora-preferences",
            title="xAI Aurora Text-to-Image Human Preferences",
            description="Human evaluation and preference rankings for text-to-image generator outputs.",
            groups=["all"],
        ),
        # Raw sources
        "raw_nano_banana": DatasetSpec(
            key="raw_nano_banana",
            path=datasets_dir / "raw" / "generator_sources" / "nano-banana",
            slug="truthlens-raw-nano-banana",
            title="TruthLens Raw Generator Source: Nano-Banana",
            description="Raw parquet partitions of nano-banana generator source.",
            groups=["raw", "all"],
        ),
        "raw_gpt_image_2": DatasetSpec(
            key="raw_gpt_image_2",
            path=datasets_dir / "raw" / "generator_sources" / "gpt-image-2",
            slug="truthlens-raw-gpt-image-2",
            title="TruthLens Raw Generator Source: GPT-Image-2",
            description="Raw generator source images from GPT-Image-2.",
            groups=["raw", "all"],
        ),
        "raw_legacy": DatasetSpec(
            key="raw_legacy",
            path=datasets_dir / "raw" / "legacy_archive",
            slug="truthlens-raw-legacy-archive",
            title="TruthLens Raw Legacy Archive",
            description="Original unflattened legacy dataset archive.",
            groups=["raw", "all"],
        ),
        "fixtures": DatasetSpec(
            key="fixtures",
            path=datasets_dir / "fixtures",
            slug="truthlens-fixtures",
            title="TruthLens Test & CI Fixtures",
            description="Smoke test fixtures and minimal evaluation sets for rapid automated testing.",
            groups=["all"],
        ),
    }

    return catalog


def format_bytes(num_bytes: int | float) -> str:
    """Format bytes into human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    b = float(num_bytes)
    for u in units:
        if b < 1024.0 or u == "TB":
            return f"{b:.2f} {u}" if u in ("MB", "GB", "TB") else f"{int(b)} {u}"
        b /= 1024.0
    return f"{b:.2f} TB"


def scan_directory(
    directory: Path, ignore_patterns: Sequence[str]
) -> tuple[int, int, int]:
    """
    Scans directory and returns (file_count, total_bytes, ignored_count).
    Matches patterns against paths relative to `directory`.
    """
    if not directory.exists():
        return 0, 0, 0

    if directory.is_file():
        return 1, directory.stat().st_size, 0

    file_count = 0
    total_bytes = 0
    ignored_count = 0

    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        for f in files:
            file_path = root_path / f
            try:
                rel_p = file_path.relative_to(directory).as_posix()
            except ValueError:
                rel_p = f

            is_ignored = any(
                fnmatch.fnmatch(rel_p, pat) or fnmatch.fnmatch(f, pat)
                for pat in ignore_patterns
            )

            if is_ignored:
                ignored_count += 1
                continue

            try:
                # Use lstat to check if symlink; if symlink, measure target if exists
                st = file_path.stat()
                total_bytes += st.st_size
                file_count += 1
            except (OSError, FileNotFoundError):
                ignored_count += 1

    return file_count, total_bytes, ignored_count


def verify_disk_space(
    target_bytes: int, file_count: int, temp_dir: Optional[Path] = None
) -> tuple[bool, str]:
    """
    Verifies that the temp directory has sufficient free disk space.
    Note: kagglehub archives directories into an uncompressed zip in tempfile when file_count > 50.
    """
    if file_count <= 50:
        return True, "Direct upload (file count <= 50, no temporary archive required)."

    check_dir = temp_dir if temp_dir else Path(tempfile.gettempdir())
    check_dir.mkdir(parents=True, exist_ok=True)

    try:
        usage = shutil.disk_usage(check_dir)
        free_bytes = usage.free
    except OSError as e:
        return True, f"Could not inspect disk usage for {check_dir}: {e}"

    # We need at least target_bytes + 1 GB safety margin
    needed_bytes = target_bytes + (1024 * 1024 * 1024)
    if free_bytes < needed_bytes:
        msg = (
            f"Insufficient disk space in temp directory ({check_dir}).\n"
            f"  Free space:     {format_bytes(free_bytes)}\n"
            f"  Required space: {format_bytes(needed_bytes)} (dataset: {format_bytes(target_bytes)} + 1 GB safety margin)\n"
            f"  Solution: specify an alternate location with '--temp-dir /path/to/larger/mount'"
        )
        return False, msg

    return True, f"Disk space OK: {format_bytes(free_bytes)} free on {check_dir} (needed ~{format_bytes(target_bytes)})."


def verify_authentication(
    username: Optional[str] = None, key: Optional[str] = None
) -> tuple[bool, str]:
    """Verifies Kaggle API credentials with kagglehub."""
    try:
        import kagglehub
    except ImportError:
        return False, "kagglehub package is not installed. Install with: pip install kagglehub"

    # If credentials supplied explicitly, configure them
    if key:
        kagglehub.config.set_kaggle_api_token(key)
    elif username and key:
        from kagglehub.config import KaggleApiCredentials, _kaggle_credentials
        import kagglehub.config as kh_config
        kh_config._kaggle_credentials = KaggleApiCredentials(username=username, key=key)

    try:
        creds = kagglehub.auth.whoami(verbose=False)
        authed_user = creds.get("username", "")
        if authed_user:
            return True, authed_user
    except Exception as e:
        return False, str(e)

    return False, "Unable to validate credentials."


def dataset_exists_on_kaggle(username: str, slug: str) -> bool:
    """Checks if a dataset already exists on Kaggle under username/slug."""
    try:
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

        with build_kaggle_client() as api:
            req = ApiGetDatasetRequest()
            req.owner_slug = username
            req.dataset_slug = slug
            api.datasets.dataset_api_client.get_dataset(req)
            return True
    except Exception:
        return False


def upload_single_dataset(
    spec: DatasetSpec,
    username: str,
    dry_run: bool = False,
    skip_existing: bool = False,
    force: bool = False,
    version_notes: str = "TruthLens dataset upload",
    temp_dir: Optional[Path] = None,
) -> bool:
    """Uploads or dry-runs a single dataset specification."""
    handle = f"{username}/{spec.slug}"
    combined_patterns = SAFE_IGNORE_PATTERNS + spec.custom_ignore

    print(f"\n{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}Dataset:{RESET}     {spec.key} ({spec.title})")
    print(f"{BOLD}Local Path:{RESET}  {spec.path}")
    print(f"{BOLD}Kaggle Handle:{RESET} {handle}")
    print(f"{CYAN}{'─' * 65}{RESET}")

    if not spec.path.exists():
        print(f"{RED}[ERROR] Local path does not exist: {spec.path}{RESET}")
        return False

    # Check remote status
    remote_exists = dataset_exists_on_kaggle(username, spec.slug)
    if remote_exists:
        print(f"  → Kaggle Status: {GREEN}Already exists remotely{RESET}")
        if skip_existing and not force:
            print(f"{GREEN}[SKIP] Dataset '{spec.slug}' already exists on Kaggle. Use --force to update.{RESET}")
            return True
    else:
        print(f"  → Kaggle Status: {YELLOW}Not found on Kaggle (will be created){RESET}")

    # Scan directory
    print(f"{YELLOW}Scanning files and calculating size...{RESET}")
    file_count, total_bytes, ignored_count = scan_directory(spec.path, combined_patterns)

    print(f"  → Active files:  {BOLD}{file_count:,}{RESET}")
    print(f"  → Total size:    {BOLD}{format_bytes(total_bytes)}{RESET}")
    print(f"  → Ignored files: {ignored_count:,} (caches/temp/secret filters)")

    if file_count == 0:
        print(f"{RED}[ERROR] No files found to upload in {spec.path}{RESET}")
        return False

    # Disk space check
    space_ok, space_msg = verify_disk_space(total_bytes, file_count, temp_dir)
    if not space_ok:
        print(f"{RED}[FATAL] {space_msg}{RESET}")
        return False
    print(f"  → {GREEN}✓ {space_msg}{RESET}")

    if dry_run:
        action_verb = "update existing" if remote_exists else "create new"
        print(f"{GREEN}[DRY-RUN] Validation passed. Would {action_verb} dataset with {file_count:,} files ({format_bytes(total_bytes)}) at {handle}{RESET}")
        return True

    # Perform upload
    import kagglehub

    old_tmpdir = os.environ.get("TMPDIR")
    if temp_dir:
        os.environ["TMPDIR"] = str(temp_dir)
        tempfile.tempdir = str(temp_dir)

    start_time = time.time()
    try:
        print(f"\n{YELLOW}Initiating upload via kagglehub...{RESET}")
        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=str(spec.path),
            version_notes=version_notes,
            ignore_patterns=combined_patterns,
        )
        elapsed = time.time() - start_time
        print(f"\n{GREEN}✔ Successfully uploaded {spec.key} to https://www.kaggle.com/datasets/{handle} in {elapsed:.1f}s{RESET}")
        return True
    except Exception as e:
        print(f"\n{RED}[ERROR] Upload failed for {spec.key}: {e}{RESET}")
        return False
    finally:
        if temp_dir:
            if old_tmpdir is not None:
                os.environ["TMPDIR"] = old_tmpdir
            else:
                os.environ.pop("TMPDIR", None)
            tempfile.tempdir = None


def print_catalog_table(catalog: dict[str, DatasetSpec]) -> None:
    """Prints a formatted summary of available datasets."""
    print(f"\n{CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║                    TruthLens — Kaggle Dataset Catalog                                ║{RESET}")
    print(f"{CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{BOLD}{'Key':<18} {'Slug':<34} {'Size':<10} {'Status':<10} {'Groups'}{RESET}")
    print(f"{'─' * 86}")

    for key, spec in catalog.items():
        exists = spec.path.exists()
        status = f"{GREEN}Available{RESET}" if exists else f"{RED}Missing{RESET}"
        if exists:
            # Quick stat
            _, size_b, _ = scan_directory(spec.path, SAFE_IGNORE_PATTERNS)
            size_str = format_bytes(size_b)
        else:
            size_str = "0 B"

        groups_str = ", ".join(spec.groups)
        print(f"{key:<18} {spec.slug:<34} {size_str:<10} {status:<19} {groups_str}")

    print(f"{'─' * 86}")
    print("Predefined groups: core, prepared, raw, all\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload TruthLens datasets to Kaggle modularly using kagglehub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    action_group = parser.add_argument_group("Selection Options")
    action_group.add_argument(
        "--list", action="store_true", help="List all cataloged datasets and their status."
    )
    action_group.add_argument(
        "--dataset", "-d", action="append", help="Specific dataset key(s) to upload (can be repeated)."
    )
    action_group.add_argument(
        "--group", "-g", choices=["core", "prepared", "raw", "all"],
        help="Upload a predefined group of datasets."
    )
    action_group.add_argument(
        "--dir", type=Path, help="Upload an arbitrary local directory (must pair with --slug)."
    )
    action_group.add_argument(
        "--slug", type=str, help="Dataset slug on Kaggle (used with --dir)."
    )

    cfg_group = parser.add_argument_group("Configuration & Safety")
    cfg_group.add_argument(
        "--dry-run", action="store_true", help="Inspect and validate without performing actual upload."
    )
    cfg_group.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip dataset if it already exists on Kaggle (default: True)."
    )
    cfg_group.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Do not skip existing datasets; upload a new version."
    )
    cfg_group.add_argument(
        "--force", "-f", action="store_true",
        help="Force upload/update even if dataset already exists."
    )
    cfg_group.add_argument(
        "--username", "-u", type=str, help="Kaggle username (defaults to auto-detection from credentials)."
    )
    cfg_group.add_argument(
        "--token", "-t", type=str, help="Kaggle API token (KGAT_... or api key)."
    )
    cfg_group.add_argument(
        "--notes", "-m", type=str, default="TruthLens dataset upload", help="Dataset version notes."
    )
    cfg_group.add_argument(
        "--temp-dir", type=Path, help="Custom directory for temporary zip generation (if /tmp is full)."
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = build_catalog()

    if args.list:
        print_catalog_table(catalog)
        return 0

    # Determine targets to process
    targets: List[DatasetSpec] = []

    if args.dir:
        if not args.slug:
            print(f"{RED}[ERROR] --dir requires --slug <dataset_slug>{RESET}")
            return 1
        spec = DatasetSpec(
            key="custom",
            path=args.dir.resolve(),
            slug=args.slug,
            title=args.slug.replace("-", " ").title(),
            description="Custom TruthLens dataset upload.",
        )
        targets.append(spec)
    elif args.dataset:
        for d in args.dataset:
            if d not in catalog:
                print(f"{RED}[ERROR] Unknown dataset '{d}'. Run --list to see available keys.{RESET}")
                return 1
            targets.append(catalog[d])
    elif args.group:
        for spec in catalog.values():
            if args.group in spec.groups:
                targets.append(spec)
    else:
        print(f"{YELLOW}[NOTICE] No dataset or group specified. Showing dataset catalog:{RESET}")
        print_catalog_table(catalog)
        print("Usage:")
        print("  python scripts/upload_datasets_to_kaggle.py --group core --dry-run")
        print("  python scripts/upload_datasets_to_kaggle.py --dataset modern_v2")
        return 0

    # Deduplicate targets while preserving order
    seen = set()
    unique_targets = []
    for t in targets:
        if t.key not in seen:
            seen.add(t.key)
            unique_targets.append(t)
    targets = unique_targets

    # Check Authentication
    print(f"\n{YELLOW}[1/2] Verifying Kaggle authentication...{RESET}")
    auth_ok, username_or_err = verify_authentication(args.username, args.token)
    if not auth_ok:
        print(f"{RED}[ERROR] Kaggle authentication failed: {username_or_err}{RESET}")
        print("\nPlease ensure your Kaggle credentials are configured:")
        print("  1. ~/.kaggle/kaggle.json exists with {\"username\": \"...\", \"key\": \"...\"}")
        print("  2. Or export KAGGLE_USERNAME=\"...\" and KAGGLE_KEY=\"...\"")
        print("  3. Or pass --username <user> --token <token>")
        return 1

    username = args.username or username_or_err
    print(f"{GREEN}  ✓ Authenticated as: {BOLD}{username}{RESET}")

    # Process targets
    print(f"\n{YELLOW}[2/2] Processing {len(targets)} dataset target(s)...{RESET}")
    successes = []
    failures = []

    for idx, spec in enumerate(targets, 1):
        print(f"\n{BOLD}[{idx}/{len(targets)}]{RESET} Processing: {spec.key}")
        ok = upload_single_dataset(
            spec=spec,
            username=username,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            force=args.force,
            version_notes=args.notes,
            temp_dir=args.temp_dir,
        )
        if ok:
            successes.append(spec.key)
        else:
            failures.append(spec.key)

    # Summary
    print(f"\n{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}Summary:{RESET}")
    print(f"  Success: {GREEN}{len(successes)}/{len(targets)}{RESET} ({', '.join(successes) if successes else 'None'})")
    if failures:
        print(f"  Failed:  {RED}{len(failures)}/{len(targets)}{RESET} ({', '.join(failures)})")
        return 1

    print(f"{GREEN}All operations completed successfully.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
