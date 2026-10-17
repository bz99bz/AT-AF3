"""Permanent post-cutoff test-only isolation guard.

The seven post-cutoff benchmark entities are never allowed in training,
validation, calibration, normalization fitting, model selection, checkpoint
selection, or sequence-clustering reference pools. This module is deliberately
small and dependency-light so every training entrypoint can import it.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DENYLIST = ROOT / "config" / "permanent_test_denylist.json"
FALLBACK_DENYLIST = ROOT / "configs" / "permanent_test_denylist.json"
PERMANENT_CASE_TABLE = ROOT / "data" / "permanent_postcutoff_test_cases.csv"
BUILTIN_DENYLIST = {
    "permanent_test_only": [
        "7P5R_2",
        "7QDI_1",
        "8AVR_2",
        "8BFD_2",
        "8GQP_1",
        "8JH6_1",
        "8P6Q_2",
    ],
    "role": "permanent_postcutoff_test",
    "source_split": "postcutoff_test",
    "is_permanent_test": True,
    "allowed_for_training": False,
    "aliases": {
        "7P5R_2": ["7P5R", "case_01_7P5R_2"],
        "7QDI_1": ["7QDI", "case_02_7QDI_1"],
        "8AVR_2": ["8AVR", "case_03_8AVR_2"],
        "8BFD_2": ["8BFD", "case_07_8BFD_2"],
        "8GQP_1": ["8GQP", "case_08_8GQP_1"],
        "8JH6_1": ["8JH6", "case_09_8JH6_1"],
        "8P6Q_2": ["8P6Q", "case_10_8P6Q_2"],
    },
}
TRAINING_PROHIBITED_SPLITS = {
    "train",
    "training",
    "validation",
    "val",
    "cross_validation",
    "cross-validation",
    "cv",
    "calibration",
    "feature_selection",
    "normalization_fit",
    "hyperparameter_search",
    "loss_weight_selection",
    "early_stopping",
    "checkpoint_selection",
    "sequence_clustering_reference_pool",
}
ALLOWED_TEST_OPERATIONS = {"transform", "predict", "evaluate"}
FORBIDDEN_TEST_OPERATIONS = {"fit", "fit_transform", "partial_fit", "calibrate", "update"}


class PermanentTestLeakageError(RuntimeError):
    """Raised when permanent test-only cases appear in a fitting input."""


if TYPE_CHECKING:
    import pandas as pd


def _optional_pandas():
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError:
        return None
    return pd


def _is_dataframe(value: object) -> bool:
    pd = _optional_pandas()
    return bool(pd is not None and isinstance(value, pd.DataFrame))


def _notna(value: object) -> bool:
    pd = _optional_pandas()
    if pd is not None:
        return bool(pd.notna(value))
    return value is not None and str(value).lower() != "nan"


def load_denylist(path: str | Path | None = None) -> dict:
    deny_path = Path(path) if path else DEFAULT_DENYLIST
    if not deny_path.exists():
        deny_path = FALLBACK_DENYLIST
    if not deny_path.exists():
        return dict(BUILTIN_DENYLIST)
    return json.loads(deny_path.read_text(encoding="utf-8"))


def _clean_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")


def _entity_from_case_id(value: object) -> str:
    text = _clean_token(value)
    match = re.search(r"([0-9][A-Z0-9]{3})_([0-9]+)", text)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return text


def permanent_alias_map(denylist: dict | None = None) -> dict[str, str]:
    deny = denylist or load_denylist()
    alias_to_entity: dict[str, str] = {}
    for entity in deny.get("permanent_test_only", []):
        canonical = _clean_token(entity)
        pdb = canonical.split("_")[0]
        aliases = {canonical, pdb, _entity_from_case_id(entity)}
        for alias in deny.get("aliases", {}).get(entity, []):
            aliases.add(_clean_token(alias))
            aliases.add(_entity_from_case_id(alias))
        for alias in aliases:
            if alias:
                alias_to_entity[alias] = canonical
    return alias_to_entity


def normalize_case_alias(value: object, denylist: dict | None = None) -> str:
    aliases = permanent_alias_map(denylist)
    clean = _clean_token(value)
    entity = _entity_from_case_id(value)
    if entity in aliases:
        return aliases[entity]
    if clean in aliases:
        return aliases[clean]
    pdb_match = re.search(r"([0-9][A-Z0-9]{3})", clean)
    if pdb_match and pdb_match.group(1) in aliases:
        return aliases[pdb_match.group(1)]
    return entity or clean


def is_permanent_test_case(value: object, denylist: dict | None = None) -> bool:
    aliases = permanent_alias_map(denylist)
    norm = normalize_case_alias(value, denylist)
    return norm in set(aliases.values())


def find_permanent_test_violations(
    records: Iterable[dict] | "pd.DataFrame",
    *,
    context: str,
    denylist: dict | None = None,
    source_path: str | Path | None = None,
) -> list[dict]:
    deny = denylist or load_denylist()
    if _is_dataframe(records):
        iterable = records.to_dict("records")
    else:
        iterable = list(records)
    violations = []
    for idx, row in enumerate(iterable):
        candidates = []
        for key in (
            "pdb_entity_id",
            "source_pdb_entity_id",
            "entity_id",
            "case_id",
            "source_case_id",
            "canonical_case_id",
            "parent_case_id",
            "pdb_id",
            "sequence_cluster_id",
            "sequence_cluster_members",
            "checkpoint_source_case_ids",
        ):
            if key in row and _notna(row[key]):
                if key == "entity_id" and "pdb_id" in row and _notna(row["pdb_id"]):
                    candidates.append(f"{row['pdb_id']}_{row[key]}")
                candidates.append(row[key])
        if source_path:
            candidates.append(str(source_path))
        for value in candidates:
            if is_permanent_test_case(value, deny):
                violations.append({
                    "context": context,
                    "source_path": str(source_path or ""),
                    "row_index": idx,
                    "matched_value": str(value),
                    "canonical_entity": normalize_case_alias(value, deny),
                    "case_id": str(row.get("case_id", "")),
                    "split": str(row.get("split", row.get("source_split", ""))),
                })
                break
    return violations


def write_violation_log(violations: list[dict], path: str | Path | None = None) -> None:
    if not violations:
        return
    out = Path(path) if path else ROOT / "reports" / "permanent_test_leakage_violations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["context", "source_path", "row_index", "matched_value", "canonical_entity", "case_id", "split"])
        writer.writeheader()
        writer.writerows(violations)


def assert_no_permanent_test_cases(
    records: Iterable[dict] | "pd.DataFrame",
    *,
    context: str,
    source_path: str | Path | None = None,
    denylist_path: str | Path | None = None,
) -> None:
    deny = load_denylist(denylist_path)
    violations = find_permanent_test_violations(records, context=context, denylist=deny, source_path=source_path)
    if violations:
        write_violation_log(violations)
        preview = "; ".join(f"{v['canonical_entity']} from {v['matched_value']} row {v['row_index']}" for v in violations[:5])
        raise PermanentTestLeakageError(
            f"Permanent post-cutoff test-only leakage blocked in {context}: {preview}. "
            f"No checkpoint or fitted artifact should be saved."
        )


def add_provenance_columns(df: "pd.DataFrame", split_col: str = "split") -> "pd.DataFrame":
    pd = _optional_pandas()
    if pd is None:
        raise ModuleNotFoundError("pandas is required for add_provenance_columns")
    out = df.copy()
    if "source_case_id" not in out.columns:
        out["source_case_id"] = out.get("case_id", "")
    splits = out[split_col].astype(str) if split_col in out.columns else pd.Series(["unassigned"] * len(out), index=out.index)
    out["source_split"] = splits.replace({"post_cutoff_test": "postcutoff_test", "post-cutoff_test": "postcutoff_test"})
    out["is_permanent_test"] = out.apply(
        lambda r: any(is_permanent_test_case(r.get(k, "")) for k in ("pdb_entity_id", "source_case_id", "case_id", "pdb_id")),
        axis=1,
    )
    out["allowed_for_training"] = (~out["is_permanent_test"]) & out["source_split"].isin(["train", "training"])
    out.loc[out["is_permanent_test"], "source_split"] = "postcutoff_test"
    out.loc[out["is_permanent_test"], "allowed_for_training"] = False
    return out


def assert_operation_allowed(case_value: object, operation: str) -> None:
    op = str(operation).strip().lower()
    if is_permanent_test_case(case_value) and op in FORBIDDEN_TEST_OPERATIONS:
        raise PermanentTestLeakageError(f"Operation '{operation}' is forbidden for permanent test-only case {case_value}")
