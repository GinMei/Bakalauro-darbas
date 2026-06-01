"""copy_assets.py

Standalone script: copy only engine-neutral asset files from a Unity project
into a target directory, preserving the relative folder structure.

Scanning scope
--------------
ONLY the "Assets/" subfolder of the Unity project is scanned.
All other root-level folders (Library/, Logs/, Packages/, ProjectSettings/,
UserSettings/, …) are completely ignored.

Within Assets/, the following direct sub-folders are excluded recursively:
    Assets/Settings/
    Assets/TutorialInfo/

Only files with explicitly approved extensions are copied.  Everything else
(Unity-specific files, hidden files, unknown types) is skipped.

Usage
-----
    python copy_assets.py <source_dir> <target_dir> [--overwrite] [--dry-run] [--verbose]

Arguments
---------
    source_dir   Root of the Unity project (the folder that CONTAINS Assets/).
    target_dir   Destination folder (created if it does not exist).
                 MUST NOT be inside source_dir/Assets — that would cause the
                 script to copy its own output, creating duplicates.
    --overwrite  Allow overwriting files that already exist in target_dir.
                 Default: existing files are skipped (safe mode).
    --dry-run    Print what would be copied without actually copying anything.
    --verbose    Also log every skipped file/folder, not just copied ones.

Exit codes
----------
    0   All eligible files copied (or skipped) successfully.
    1   One or more files could not be copied (errors printed to stderr).
    2   Bad arguments / source directory not found / self-copy detected.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Filtering configuration
# ---------------------------------------------------------------------------

# Only the Assets/ folder is scanned.  All other root-level Unity folders are
# ignored without ever opening them.
_ASSETS_FOLDER = "Assets"

# Direct children of Assets/ to skip entirely (along with all their contents).
# These are Unity template/configuration folders that have no use in Godot.
ASSETS_EXCLUDED_SUBDIRS: frozenset[str] = frozenset({
    "Settings",      # Unity render-pipeline / project settings assets
    "TutorialInfo",  # Unity template tutorial files — not game content
})

# Whitelist: only these extensions are ever copied.
# Any file whose extension is not in this set is silently skipped.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    # 3-D models — portable interchange formats
    ".fbx",   # Autodesk FBX: widely supported by both engines
    ".obj",   # Wavefront OBJ: plain-text mesh, no engine dependency
    # Textures / images
    ".png",   # Lossless raster; preferred for game textures
    ".jpg",   # Lossy raster; common for environment maps / photos
    ".jpeg",  # Alias for .jpg
    # Audio
    ".wav",   # Uncompressed PCM; universally supported
    ".mp3",   # Lossy compressed; Godot AudioStreamMP3
    ".ogg",   # Vorbis; native to Godot, supported by Unity
    # Fonts
    ".ttf",   # TrueType; cross-platform
    ".otf",   # OpenType; cross-platform
    # Generic engine-neutral data
    ".json",  # Structured data; no engine-specific schema
    ".txt",   # Plain text; dialogue, config, notes
})


# ---------------------------------------------------------------------------
# Logger — callers can configure this; CLI sets the level via --verbose
# ---------------------------------------------------------------------------

log = logging.getLogger("copy_assets")


# ---------------------------------------------------------------------------
# Core walk logic
# ---------------------------------------------------------------------------

def iter_allowed_files(unity_project_root: Path):
    """Walk Assets/ and yield (absolute_src_path, path_relative_to_project_root).

    The yielded relative path always starts with "Assets/…", so the target
    directory will mirror the Unity project structure.

    Directories are pruned before descent:
      - Hidden dirs (name starts with '.')  → skipped
      - Names in ASSETS_EXCLUDED_SUBDIRS (only at the first level of Assets/)
        → skipped with a log message

    Files are skipped when:
      - Name starts with '.'  (hidden / system file)
      - Extension not in ALLOWED_EXTENSIONS
    """
    assets_dir = unity_project_root / _ASSETS_FOLDER
    if not assets_dir.is_dir():
        log.warning("Assets/ folder not found under: %s", unity_project_root)
        return

    for dirpath, dirnames, filenames in os.walk(assets_dir):
        current = Path(dirpath)
        is_assets_root = (current == assets_dir)

        # ── Prune subdirectories before os.walk descends into them ──────────
        keep: list[str] = []
        for d in dirnames:
            if d.startswith("."):
                log.debug("skip hidden dir  : %s", current / d)
                continue
            if is_assets_root and d in ASSETS_EXCLUDED_SUBDIRS:
                log.info("skip excluded dir: %s", current / d)
                continue
            keep.append(d)
        dirnames[:] = keep  # mutate in-place — os.walk respects this

        # ── Yield whitelisted files in the current directory ────────────────
        for fname in filenames:
            path = current / fname

            if fname.startswith("."):
                log.debug("skip hidden file : %s", path)
                continue

            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                log.debug("skip non-allowed : %s", path)
                continue

            try:
                rel = path.relative_to(unity_project_root)
            except ValueError:
                log.warning("skip (bad rel)   : %s", path)
                continue

            yield path, rel


def _check_no_self_copy(assets_dir: Path, target_dir: Path) -> bool:
    """Return True when target_dir is inside assets_dir (self-copy detected).

    Copying INTO the Assets tree being scanned would cause the walk to visit
    previously-copied output files, leading to duplicates and possible loops.
    """
    try:
        target_dir.relative_to(assets_dir)
        return True   # target is inside assets_dir
    except ValueError:
        return False  # target is outside — safe


def copy_assets(
    unity_project_root: Path,
    target_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[int, int, list[str]]:
    """Copy allowed asset files from Assets/ into target_dir.

    Returns:
        (copied_count, skipped_count, error_messages)
    """
    assets_dir = unity_project_root / _ASSETS_FOLDER

    if _check_no_self_copy(assets_dir, target_dir):
        msg = (
            f"target_dir ({target_dir}) is inside the Assets/ folder being "
            f"scanned ({assets_dir}). This would cause recursive duplication. "
            f"Choose a target outside the Unity project."
        )
        return 0, 0, [f"FATAL: {msg}"]

    copied  = 0
    skipped = 0
    errors: list[str] = []

    for src, rel in iter_allowed_files(unity_project_root):
        dst = target_dir / rel

        if dst.exists() and not overwrite:
            log.debug("skip (exists)    : %s", dst)
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            log.info("copied           : %s", rel)
            copied += 1
        except OSError as exc:
            msg = f"ERROR copying {rel}: {exc}"
            errors.append(msg)
            log.error(msg)

    return copied, skipped, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy engine-neutral assets from a Unity project's Assets/ folder "
            "into a target directory, preserving the folder structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source_dir",
        help="Unity project root — the folder that CONTAINS Assets/ (not Assets/ itself).",
    )
    parser.add_argument(
        "target_dir",
        help="Destination folder (created if absent). Must not be inside source_dir/Assets/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite files that already exist in target_dir. Default: skip existing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List what would be copied without actually copying anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Log every skipped file and folder (in addition to copied files).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)-8s %(message)s",
        level=level,
        stream=sys.stdout,
    )

    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()

    # ── Validate source ──────────────────────────────────────────────────────
    if not source_dir.exists():
        log.error("source directory not found: %s", source_dir)
        return 2
    if not source_dir.is_dir():
        log.error("source path is not a directory: %s", source_dir)
        return 2

    assets_dir = source_dir / _ASSETS_FOLDER
    if not assets_dir.is_dir():
        log.error(
            "Assets/ folder not found inside source_dir.\n"
            "  source_dir should be the UNITY PROJECT ROOT, not the Assets/ folder itself.\n"
            "  Expected: %s",
            assets_dir,
        )
        return 2

    # ── Self-copy guard ──────────────────────────────────────────────────────
    if _check_no_self_copy(assets_dir, target_dir):
        log.error(
            "target_dir is inside Assets/ — this would copy files into the folder "
            "currently being scanned, causing recursive duplication.\n"
            "  assets_dir : %s\n"
            "  target_dir : %s",
            assets_dir, target_dir,
        )
        return 2

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("[dry-run] source : %s", source_dir)
        log.info("[dry-run] target : %s", target_dir)
        count = 0
        for _, rel in iter_allowed_files(source_dir):
            log.info("[dry-run] would copy: %s", rel)
            count += 1
        log.info("[dry-run] %d file(s) would be copied.", count)
        return 0

    # ── Copy ─────────────────────────────────────────────────────────────────
    target_dir.mkdir(parents=True, exist_ok=True)
    copied, skipped, errors = copy_assets(source_dir, target_dir, overwrite=args.overwrite)

    log.info(
        "Done.  Copied: %d  |  Skipped (already exist): %d  |  Errors: %d",
        copied, skipped, len(errors),
    )
    for err in errors:
        log.error(err)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
