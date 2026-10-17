#!/usr/bin/env python3
"""Check the interpreter and optional external tools used by AT-AF3.

This script is intentionally non-destructive. It reports what is available and
returns a non-zero status only when a Python package required by the released
code is missing. AF3, Phenix, DockQ and USalign are optional at this stage; the
pipeline scripts document where each is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("numpy", "pandas", "gemmi", "matplotlib")


def check_python() -> bool:
    ok = sys.version_info >= (3, 10)
    print(f"python: {sys.executable} ({sys.version.split()[0]}) {'OK' if ok else 'REQUIRES >=3.10'}")
    return ok


def check_packages() -> bool:
    ok = True
    for name in REQUIRED:
        found = importlib.util.find_spec(name) is not None
        print(f"package {name}: {'OK' if found else 'MISSING'}")
        ok &= found
    return ok


def check_path(label: str, value: str | None) -> bool:
    if not value:
        print(f"{label}: not configured")
        return False
    path = Path(value)
    if path.is_file() or path.is_dir():
        print(f"{label}: {path} OK")
        return True
    resolved = shutil.which(value)
    if resolved:
        print(f"{label}: {resolved} OK")
        return True
    print(f"{label}: {value} NOT FOUND")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-external", action="store_true", help="fail when AF3/Phenix/DockQ/USalign are absent")
    args = parser.parse_args(argv)

    ok = check_python() and check_packages()
    af3_root = os.environ.get("AF3_ROOT")
    external = [
        check_path("AF3 root", af3_root),
        check_path("AF3 Python", os.environ.get("AF3_PYTHON")),
        check_path("AF3 model", os.environ.get("AF3_MODEL_DIR")),
        check_path("Phenix Python", os.environ.get("PHENIX_PYTHON", "phenix.python")),
        check_path("Phenix MolProbity", os.environ.get("PHENIX_MOLPROBITY", "phenix.molprobity")),
        check_path("Phenix reduce", os.environ.get("PHENIX_REDUCE", "phenix.reduce")),
        check_path("DockQ", os.environ.get("DOCKQ", "DockQ")),
        check_path("USalign", os.environ.get("USALIGN", str(ROOT / "bin/USalign"))),
    ]
    if args.strict_external:
        ok &= all(external)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
