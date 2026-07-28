"""Stage the static half of the browser bundle in ``dist/``.

``npm run build`` runs this first and then ``tsc``: this script wipes ``dist/``
and copies in what is not compiled — the page, the stylesheet, the icon, and
the Python modules the Pyodide worker imports, which land in ``pysrc/``.

Keeping the Python payload inside the site root is what lets every URL in the
app stay relative, so one bundle works both at a domain root and under a
project-pages path.

This script deliberately uses nothing outside the standard library, so CI and
``mise run build`` can run it with any Python 3.11+ instead of syncing the
project's PySide6 environment first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = REPO_ROOT / "web"
PACKAGE_NAME = "game_of_life_text"
PACKAGE_ROOT = REPO_ROOT / "src" / PACKAGE_NAME

SITE_FILES = ("index.html", "styles.css", "favicon.svg")

# The planner and its dependencies, and nothing else: gui.py and the profiling
# entry points import PySide6 and line-profiler, which Pyodide has no wheels
# for. tests/test_build_web.py checks this list against what browser.py and its
# imports actually reach, so a new module cannot silently go missing here.
WEB_MODULES = (
    "__init__.py",
    "browser.py",
    "construction.py",
    "font.py",
    "simulator.py",
    "text.py",
)


def build(out_dir: Path) -> list[Path]:
    """Write the static bundle to ``out_dir`` and return the files it contains."""

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    written: list[Path] = []
    for name in SITE_FILES:
        shutil.copy2(WEB_ROOT / name, out_dir / name)
        written.append(out_dir / name)

    package_dir = out_dir / "pysrc" / PACKAGE_NAME
    package_dir.mkdir(parents=True)
    for name in WEB_MODULES:
        shutil.copy2(PACKAGE_ROOT / name, package_dir / name)
        written.append(package_dir / name)

    # The worker reads this instead of hardcoding the module list, so the copy
    # above stays the single source of truth for what ships to the browser.
    manifest = out_dir / "pysrc" / "manifest.json"
    manifest.write_text(
        json.dumps({"package": PACKAGE_NAME, "modules": list(WEB_MODULES)}, indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(manifest)
    return written


def main(argv: list[str] | None = None) -> int:
    """Build the static bundle from the command line."""

    parser = argparse.ArgumentParser(description="Stage the static browser bundle.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "dist",
        help="output directory (default: dist/)",
    )
    args = parser.parse_args(argv)

    written = build(args.out.resolve())
    total = sum(path.stat().st_size for path in written)
    print(f"{args.out}: {len(written)} files, {total / 1024:.0f} KiB (tsc adds the modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
