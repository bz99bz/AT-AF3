#!/usr/bin/env python
"""Interaction preservation across GT / AF3 / uncorrected AT-AF3 / corrected AT-AF3.

Phenix/iotbx implementation.  Run it with the Phenix python so that ``iotbx``/``cctbx`` are
importable::

    source /path/to/phenix/phenix_env.sh
    phenix.python scripts/phenix_interaction_analysis.py --cases 7P5R_2 --max-seeds 2

Deliverables land in ``--out-dir`` (``results/phenix_interaction`` by default):
``manifest.csv``, ``residue_mapping.csv``, ``preparation_audit.csv``, ``interactions.csv``,
the two comparison CSVs, ``chemistry_audit.csv``, ``tool_inventory.csv`` and ``audit.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phenix_interaction_analysis import protocol as P  # noqa: E402
from phenix_interaction_analysis import run as RUN  # noqa: E402


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=None, help="subset of target ids")
    ap.add_argument("--max-seeds", type=int, default=0, help="keep the first N seeds (0 = all)")
    ap.add_argument("--out-dir", type=Path, default=RUN.DEFAULT_OUT)
    ap.add_argument("--workdir", type=Path, default=RUN.ROOT / "results/phenix_interaction/work")
    ap.add_argument("--canonical-dir", type=Path, default=RUN.DEFAULT_CANONICAL)
    ap.add_argument("--chain-map", type=Path, default=RUN.DEFAULT_CHAIN_MAP)
    ap.add_argument("--chirality-dir", type=Path, default=RUN.DEFAULT_CHIRALITY)
    ap.add_argument("--ccd-cache", type=Path, default=RUN.DEFAULT_CCD_CACHE)
    ap.add_argument("--reduce", default=RUN.DEFAULT_REDUCE, help="path to phenix.reduce")
    ap.add_argument("--no-reduce", action="store_true",
                    help="skip hydrogen placement (hbonds become polar contacts)")
    ap.add_argument("--keep-all-altlocs", action="store_true",
                    help="keep every altloc instead of the highest-occupancy one")
    ap.add_argument("--show-protocol", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)
    if args.show_protocol:
        print(json.dumps(P.PROTOCOL_CHOICES, indent=1, default=str))
        print(json.dumps({"provenance": P.PROVENANCE_NOTES}, indent=1))
    settings = RUN.Settings(
        canonical_dir=args.canonical_dir, chain_map_json=args.chain_map,
        chirality_dir=args.chirality_dir, ccd_cache=args.ccd_cache, out_dir=args.out_dir,
        workdir=args.workdir, reduce_tool="" if args.no_reduce else str(args.reduce),
        cases=list(args.cases or []), max_seeds=args.max_seeds,
        altloc_policy="all" if args.keep_all_altlocs else "best",
        use_reduce=not args.no_reduce)
    report = RUN.run(settings, progress=lambda m: print(m, flush=True))
    print("\nwrote:")
    for name, count in sorted(report.get("deliverables", {}).items()):
        print(f"  {name:44s} {count:>8d} rows")
    print(f"  targets: {', '.join(report.get('targets', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
