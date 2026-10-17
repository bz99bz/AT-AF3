"""Binary chirality error-code construction.

The error code is derived only from the current predicted structure and the
declared target chirality config. Reference structures are intentionally absent
from this API.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .chirality_geometry import classify_residue_chirality, iter_residues


def _load_csv_target_config(path: Path, case_id: str | None) -> dict:
    if not case_id:
        raise ValueError("case_id is required when loading a CSV target manifest")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if row.get("case_id") == case_id]
    if not rows:
        raise ValueError(f"target manifest contains no rows for case_id={case_id}")

    chains: dict[str, dict] = {}
    for row in rows:
        if str(row.get("correction_supported", "true")).strip().lower() not in {"true", "1", "yes"}:
            continue
        chain_id = str(row.get("chain_id", "")).strip()
        residue_id = str(row.get("label_seq_id") or row.get("residue_id") or "").strip()
        target = str(row.get("target_chirality", "")).strip().upper()
        if not chain_id or not residue_id or target not in {"D", "L"}:
            raise ValueError(f"invalid target-manifest row for {case_id}: {row}")
        try:
            residue_id = str(int(float(residue_id)))
        except ValueError as exc:
            raise ValueError(f"invalid residue identifier for {case_id}: {residue_id}") from exc
        rule = chains.setdefault(chain_id, {
            "default_target_chirality": "ignore",
            "exclude_residue_names": ["GLY"],
            "residue_targets": {},
        })
        previous = rule["residue_targets"].get(residue_id)
        if previous and previous != target:
            raise ValueError(f"conflicting targets for {case_id} {chain_id}:{residue_id}")
        rule["residue_targets"][residue_id] = target
    if not chains:
        raise ValueError(f"target manifest has no correction-supported rows for case_id={case_id}")
    return {
        "case_id": case_id,
        "mode": "local_D",
        "chains": chains,
        "reference_coordinates_allowed": False,
        "target_source": str(path),
    }


def load_target_config(path: str | Path, case_id: str | None = None) -> dict:
    target_path = Path(path)
    if target_path.suffix.lower() == ".csv":
        config = _load_csv_target_config(target_path, case_id)
    else:
        config = json.loads(target_path.read_text(encoding="utf-8"))
    if config.get("reference_coordinates_allowed", False):
        raise ValueError("reference_coordinates_allowed must be false for chirality correction")
    if "chains" not in config or not isinstance(config["chains"], dict):
        raise ValueError("target config must define chains")
    return config


def chain_rule(config: dict, chain_id: str) -> dict:
    chains = config.get("chains", {})
    if chain_id in chains:
        return dict(chains[chain_id])
    if "*" in chains:
        return dict(chains["*"])
    return {}


def target_chirality_for_residue(config: dict, chain_id: str, residue_index: int, residue_name: str) -> str | None:
    rule = chain_rule(config, chain_id)
    excluded = set(rule.get("exclude_residue_names", []))
    if residue_name == "GLY" or residue_name in excluded:
        return "achiral"
    if residue_name == "UNK":
        return "unresolved"
    overrides = rule.get("residue_targets", {})
    if str(residue_index) in overrides:
        return overrides[str(residue_index)]
    if residue_index in overrides:
        return overrides[residue_index]
    return rule.get("default_target_chirality")


def residue_can_be_modified(config: dict, chain_id: str) -> bool:
    rule = chain_rule(config, chain_id)
    return bool(rule.get("allow_coordinate_modification", True))


def build_error_code_rows(atoms: list[dict], config: dict, case_id: str, seed: int, epsilon: float = 1e-6) -> list[dict]:
    rows = []
    for residue in iter_residues(atoms):
        chain_id = residue["chain_id"]
        resid = residue["residue_index"]
        resname = residue["residue_name"]
        target = target_chirality_for_residue(config, chain_id, resid, resname)
        meas = classify_residue_chirality(residue["atoms"], epsilon=epsilon)
        current = meas["chirality"]
        status = meas["chirality_status"]
        can_modify = residue_can_be_modified(config, chain_id)

        error_code = None
        eligible = False
        reason = ""
        if target in (None, "", "ignore"):
            reason = "no_target_chirality_declared"
        elif target == "unresolved":
            reason = "unknown_residue_identity"
        elif target == "achiral" or status == "achiral":
            reason = meas["reason"]
        elif status in {"unresolved", "ambiguous"}:
            reason = meas["reason"]
        elif target not in {"D", "L"}:
            reason = f"unsupported_target_chirality:{target}"
        elif current == target:
            error_code = 0
            reason = "matches_target"
        else:
            error_code = 1
            if can_modify:
                eligible = True
                reason = "chirality_mismatch"
            else:
                reason = "chain_coordinate_modification_forbidden"

        rows.append({
            "case_id": case_id,
            "seed": seed,
            "chain_id": chain_id,
            "residue_index": resid,
            "insertion_code": residue["insertion_code"],
            "residue_name": resname,
            "target_chirality": target if target is not None else "",
            "current_chirality": current if current is not None else "",
            "signed_volume_before": meas["signed_volume"],
            "chirality_status": status,
            "error_code": error_code,
            "correction_eligible": eligible,
            "reason_if_ineligible": "" if eligible else reason,
            "classification_reason": reason,
            "allow_coordinate_modification": can_modify,
            "reference_used_for_correction": False,
        })
    return rows


def error_code_json(rows: list[dict], case_id: str, seed: int) -> dict:
    chains: dict[str, list] = {}
    residues: dict[str, list] = {}
    for row in rows:
        chain = row["chain_id"]
        chains.setdefault(chain, []).append(row["error_code"])
        residues.setdefault(chain, []).append({
            "residue_index": row["residue_index"],
            "residue_name": row["residue_name"],
            "target_chirality": row["target_chirality"],
            "current_chirality": row["current_chirality"],
            "error_code": row["error_code"],
            "correction_eligible": row["correction_eligible"],
        })
    return {
        "case_id": case_id,
        "seed": seed,
        "chains": chains,
        "residues": residues,
        "correction_method": "binary_chirality_error_code",
        "reference_used_for_correction": False,
        "target_chirality_source": "input_condition_or_CCD",
    }


ERROR_CODE_COLUMNS = [
    "case_id",
    "seed",
    "chain_id",
    "residue_index",
    "insertion_code",
    "residue_name",
    "target_chirality",
    "current_chirality",
    "signed_volume_before",
    "chirality_status",
    "error_code",
    "correction_eligible",
    "reason_if_ineligible",
    "classification_reason",
    "allow_coordinate_modification",
    "reference_used_for_correction",
]


def write_error_code_outputs(rows: list[dict], out_dir: str | Path, case_id: str, seed: int) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "chirality_error_code.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_CODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    package = error_code_json(rows, case_id, seed)
    json_path = out / "chirality_error_code.json"
    json_path.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")
    return package
