"""Build ``manifest.csv``: one row per (target, method, seed, sample) structure.

The row carries the identity and provenance a reader needs to reproduce the analysis:

* ``target``            the benchmark target id (``7P5R_2``);
* ``task``              the target's modelling task / chirality mode from its config;
* ``method``            GT / AF3 / AT-AF3 / Corrected AT-AF3;
* ``seed``, ``sample``  the inference seed and the AF3 sample index;
* ``structure_path``    the canonical PDB that is read;
* ``reference_path``    the GT PDB the method is compared against;
* ``selected_chains``   the chains selected for this model against the config;
* plus ``dataset``, ``condition_suffix``, ``source_path``, ``sha256``, ``exists`` and
  ``expected_cofactors``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import protocol as P

FILE_RE = re.compile(
    r"^(?P<dataset>All_D|Local_D)_(?P<case>.+?)_seed_(?P<seed>\d+)_(?P<suffix>.+)\.pdb$")
SAMPLE_RE = re.compile(r"sample-(\d+)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_label(raw: str) -> str:
    return raw.replace("_", "-")


def _task_mode(case: str, chirality_dir: Path) -> tuple:
    cfg = Path(chirality_dir) / f"{case}.json"
    if not cfg.exists():
        return P.NA, ""
    try:
        data = json.loads(cfg.read_text())
    except Exception:
        return P.NA, ""
    mode = data.get("mode", P.NA)
    chains = ",".join(sorted((data.get("chains") or {}).keys()))
    return mode, chains


def build_manifest(canonical_dir, chain_map_json, chirality_dir) -> list:
    canonical_dir = Path(canonical_dir)
    mapping = {}
    if Path(chain_map_json).exists():
        mapping = json.loads(Path(chain_map_json).read_text())
    parsed = []
    gt_index = {}
    for f in sorted(canonical_dir.glob("*.pdb")):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        d = m.groupdict()
        if d["suffix"] == "GT":
            gt_index[(d["dataset"], d["case"])] = f
        parsed.append((d, f))

    rows = []
    for d, f in parsed:
        method = P.SUFFIX_TO_METHOD.get(d["suffix"], d["suffix"])
        ref = gt_index.get((d["dataset"], d["case"]), f if method == P.METHOD_GT else None)
        entry = mapping.get(f.name, {}) or {}
        source = entry.get("source", "")
        cmap = entry.get("chain_id_map", {}) or {}
        sample = P.NA
        ms = SAMPLE_RE.search(source)
        if ms:
            sample = ms.group(1)
        mode, cfg_chains = _task_mode(d["case"], Path(chirality_dir))
        if method == P.METHOD_GT:
            selected = cfg_chains or ",".join(sorted(set(cmap.values())))
        else:
            selected = ",".join(sorted(set(cmap.values()))) or cfg_chains
        rows.append({
            "target": d["case"],
            "task": mode,
            "method": method,
            "seed": d["seed"],
            "sample": sample,
            "structure_path": str(f),
            "reference_path": str(ref) if ref else P.NA,
            "selected_chains": selected,
            "dataset": _dataset_label(d["dataset"]),
            "condition_suffix": d["suffix"],
            "source_path": source or P.NA,
            "sha256": _sha256(f),
            "exists": "yes",
            "expected_cofactors": "",
            "notes": "",
        })
    return rows
