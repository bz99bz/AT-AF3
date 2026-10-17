from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
import shlex
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOCAL_D_ROOT = ROOT / "data" / "local_d"
OUT_DIR = LOCAL_D_ROOT / "local_d_curated_v1"
VERSION = "local_d_curated_v1"
CUTOFF = dt.date(2021, 9, 30)
PERM_CASES = {"7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"}
PERM_PDBS = {x.split("_", 1)[0] for x in PERM_CASES}
OLD_SEEDS = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010}

FILES = {
    "candidates": "pdb_local_d_candidates.csv",
    "review": "pdb_local_d_review.csv",
    "all_d_like_excluded": "pdb_all_d_like_excluded_from_local.csv",
    "false_positive_excluded": "pdb_local_d_false_positive_excluded.csv",
    "all_evidence": "pdb_local_d_all_evidence_rows.csv",
    "split_before": "pdb_local_d_candidates_before_or_on_2021-09-30.csv",
    "split_after": "pdb_local_d_candidates_after_2021-09-30.csv",
    "split_unknown": "pdb_local_d_candidates_release_date_unknown.csv",
}
REQ = [
    "pdb_id", "entity_id", "auth_chains", "asym_chains", "polymer_type",
    "entity_description", "entry_title", "initial_release_date", "deposit_date",
    "length", "non_gly_length", "direct_d_residue_count", "d_fraction_excluding_gly",
    "d_residue_positions_one_based", "d_residue_ccd_codes", "d_chem_comp_monomers",
    "is_direct_all_d_chain", "local_d_verdict", "confidence", "d_detection_source",
    "search_source_tags", "matched_d_terms", "sequence_monomers", "sequence_one_letter",
]
D_CODES = {
    "DAL", "DAR", "DSN", "DSG", "DAS", "DCY", "DGN", "DGL", "DHI", "DIL",
    "DLE", "DLY", "DME", "MED", "DPN", "DPR", "DSE", "DTH", "DTR", "DTY", "DVA",
    "28J", "HJV", "HJY", "ZAE", "MLU", "A1EJL", "A1EQM", "A1CTT",
}
D_NAME = {
    "ALA": "DAL", "ALANINE": "DAL", "ARG": "DAR", "ARGININE": "DAR",
    "ASN": "DSN", "ASPARAGINE": "DSN", "ASP": "DAS", "ASPARTATE": "DAS", "ASPARTIC": "DAS",
    "CYS": "DCY", "CYSTEINE": "DCY", "GLN": "DGN", "GLUTAMINE": "DGN",
    "GLU": "DGL", "GLUTAMATE": "DGL", "GLUTAMIC": "DGL", "HIS": "DHI", "HISTIDINE": "DHI",
    "ILE": "DIL", "ISOLEUCINE": "DIL", "LEU": "DLE", "LEUCINE": "DLE",
    "LYS": "DLY", "LYSINE": "DLY", "MET": "MED", "METHIONINE": "MED",
    "PHE": "DPN", "PHENYLALANINE": "DPN", "PRO": "DPR", "PROLINE": "DPR",
    "SER": "DSE", "SERINE": "DSE", "THR": "DTH", "THREONINE": "DTH",
    "TRP": "DTR", "TRYPTOPHAN": "DTR", "TYR": "DTY", "TYROSINE": "DTY",
    "VAL": "DVA", "VALINE": "DVA",
}
TEXT_D = re.compile(r"D\s*[- ]\s*(" + "|".join(sorted(D_NAME, key=len, reverse=True)) + r")", re.I)
ALL_D_TEXT = re.compile(r"\b(ALL[- ]?D|D[- ]?PROTEIN|MIRROR[- ]?IMAGE|RACEMIC|QUASI[- ]?RACEMIC|D[- ]?ENANTIOMER)\b", re.I)

CASE_COLS = [
    "dataset_version", "source_csv", "source_row_id", "pdb_id_raw", "pdb_id_normalized",
    "entity_id", "case_id_normalized", "auth_chains_raw", "asym_chains_raw", "chain_id_normalized",
    "polymer_type", "entity_description", "entry_title", "initial_release_date",
    "initial_release_date_parsed", "deposit_date", "split", "final_status", "reason_code",
    "target_d_residue_count", "correction_supported_target_count", "sequence_length",
    "sequence_one_letter_canonical", "sequence_sha256", "d_ccd_codes_detected",
    "target_chirality_source", "reference_allowed_for_curation", "reference_allowed_for_evaluation",
    "reference_allowed_for_correction", "allowed_for_training", "locked_post_cutoff_candidate",
    "permanent_test_overlap", "duplicate_or_leakage_flag", "manual_review_required", "notes",
]
RES_COLS = [
    "dataset_version", "split", "pdb_id", "case_id", "model_id", "assembly_id", "entity_id",
    "chain_id", "auth_seq_id", "label_seq_id", "insertion_code", "residue_name", "ccd_id",
    "polymer_type", "target_chirality", "target_chirality_source", "is_target_d_residue",
    "is_standard_amino_acid", "is_modified_amino_acid", "has_n", "has_ca", "has_c", "has_o",
    "has_cb", "has_sidechain_atoms", "local_frame_available", "correction_supported",
    "correction_unsupported_reason", "source_csv", "source_row_id", "manual_review_required",
    "reference_observed_chirality", "reference_geometry_valid", "reference_allowed_for_curation",
    "reference_allowed_for_evaluation", "reference_allowed_for_correction", "notes",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def nstr(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def pdb(v: Any) -> str:
    return nstr(v).upper()


def cid(row: dict | pd.Series) -> str:
    return f"{pdb(row.get('pdb_id'))}_{nstr(row.get('entity_id'))}"


def key(row: dict | pd.Series) -> str:
    return f"{pdb(row.get('pdb_id'))}|{nstr(row.get('entity_id'))}"


def parse_date(v: Any) -> dt.date | None:
    s = nstr(v)
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return dt.date.fromisoformat(s[:10])
        except Exception:
            return None


def split(v: Any) -> str:
    d = parse_date(v)
    if d is None:
        return "release_date_unknown"
    return "before_or_on_cutoff" if d <= CUTOFF else "locked_post_cutoff_candidate"


def pieces(v: Any) -> list[str]:
    s = nstr(v)
    return [x.strip().upper() for x in re.split(r"[;,|]", s) if x.strip()]


def row_d_codes(row: dict | pd.Series) -> set[str]:
    out = set(pieces(row.get("d_residue_ccd_codes"))) | set(pieces(row.get("d_chem_comp_monomers")))
    for text in (row.get("entry_title", ""), row.get("entity_description", "")):
        for m in TEXT_D.finditer(nstr(text).upper()):
            out.add(D_NAME[m.group(1).upper()])
    return {x for x in out if x and x.lower() != "nan"}


def all_target_codes(row: dict | pd.Series) -> set[str]:
    return set(D_CODES) | row_d_codes(row)


def read_frames(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        k: pd.read_csv(input_dir / name, dtype=str, keep_default_na=False)
        if (input_dir / name).exists() else pd.DataFrame(columns=REQ)
        for k, name in FILES.items()
    }


def split_cif_row(line: str) -> list[str]:
    return shlex.split(line, posix=True)


def iter_loops(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        rows = []
        while i < len(lines):
            s = lines[i].strip()
            if not s or s == "#" or s == "loop_" or s.startswith("_") or s.startswith("data_"):
                break
            try:
                rows.append(split_cif_row(s))
            except Exception:
                pass
            i += 1
        yield headers, rows


def clean(v: Any) -> str:
    s = "" if v is None else str(v)
    return "" if s in {".", "?", "nan", "NaN"} else s


def to_int(v: Any) -> int | None:
    try:
        return int(float(str(v)))
    except Exception:
        return None


def atom_site(text: str) -> list[dict]:
    for headers, rows in iter_loops(text):
        if not headers or not all(h.startswith("_atom_site.") for h in headers):
            continue
        idx = {h: i for i, h in enumerate(headers)}
        if "_atom_site.label_atom_id" not in idx or "_atom_site.label_comp_id" not in idx:
            continue
        atoms = []
        for serial, r in enumerate(rows, 1):
            if len(r) < len(headers):
                continue
            model = to_int(r[idx["_atom_site.pdbx_PDB_model_num"]]) if "_atom_site.pdbx_PDB_model_num" in idx else 1
            if model is not None and model != 1:
                continue
            label_seq = to_int(r[idx["_atom_site.label_seq_id"]]) if "_atom_site.label_seq_id" in idx else None
            auth_seq = to_int(r[idx["_atom_site.auth_seq_id"]]) if "_atom_site.auth_seq_id" in idx else label_seq
            atoms.append({
                "serial": serial,
                "label_atom_id": clean(r[idx["_atom_site.label_atom_id"]]).upper(),
                "auth_atom_id": clean(r[idx["_atom_site.auth_atom_id"]]).upper() if "_atom_site.auth_atom_id" in idx else clean(r[idx["_atom_site.label_atom_id"]]).upper(),
                "label_comp_id": clean(r[idx["_atom_site.label_comp_id"]]).upper(),
                "label_asym_id": clean(r[idx["_atom_site.label_asym_id"]]) if "_atom_site.label_asym_id" in idx else "",
                "auth_asym_id": clean(r[idx["_atom_site.auth_asym_id"]]) if "_atom_site.auth_asym_id" in idx else "",
                "label_entity_id": clean(r[idx["_atom_site.label_entity_id"]]) if "_atom_site.label_entity_id" in idx else "",
                "label_seq_id": label_seq,
                "auth_seq_id": auth_seq,
                "ins_code": clean(r[idx["_atom_site.pdbx_PDB_ins_code"]]) if "_atom_site.pdbx_PDB_ins_code" in idx else "",
            })
        return atoms
    return []


def residues(atoms: list[dict]) -> list[dict]:
    grouped, order = {}, {}
    for i, a in enumerate(atoms):
        k = (a["label_entity_id"], a["label_asym_id"], a["auth_asym_id"], a["label_seq_id"], a["auth_seq_id"], a["ins_code"], a["label_comp_id"])
        grouped.setdefault(k, []).append(a)
        order.setdefault(k, i)
    out = []
    for k in sorted(grouped, key=lambda x: order[x]):
        ent, lasym, aasym, lseq, aseq, ins, comp = k
        out.append({"entity": ent, "label_asym": lasym, "auth_asym": aasym, "label_seq": lseq, "auth_seq": aseq, "ins": ins, "comp": comp, "atoms": grouped[k]})
    return out


def fetch(url: str, path: Path, allow_network: bool) -> str:
    if path.exists() and path.stat().st_size:
        return path.read_text(encoding="utf-8", errors="replace")
    if not allow_network:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AT-AF3-local-D-curated-v1"})
            with urllib.request.urlopen(req, timeout=60) as h:
                txt = h.read().decode("utf-8", errors="replace")
            path.write_text(txt, encoding="utf-8")
            return txt
        except Exception as exc:
            last = exc
            time.sleep(i + 1)
    raise RuntimeError(f"failed download {url}: {last}")


def canonical_sequence(entity_json: dict) -> str:
    ep = entity_json.get("entity_poly", {}) or {}
    seq = ep.get("pdbx_seq_one_letter_code_can") or ep.get("pdbx_seq_one_letter_code") or ""
    return re.sub(r"\s+", "", re.sub(r"\([^)]*\)", "X", str(seq))).upper()


def audit_inputs(input_dir: Path, out_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    frames = read_frames(input_dir)
    rows, hashes = [], {}
    for k, name in FILES.items():
        p = input_dir / name
        df = frames[k]
        hashes[name] = sha(p) if p.exists() else "missing"
        miss = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns}
        rows.append({
            "source_key": k, "file_name": name, "path": rel(p), "exists": p.exists(), "sha256": hashes[name],
            "n_rows": len(df), "n_columns": len(df.columns), "columns_json": json.dumps(list(df.columns), ensure_ascii=False),
            "missing_required_columns": ";".join([c for c in REQ if c not in df.columns]),
            "duplicate_full_rows": int(df.duplicated().sum()) if not df.empty else 0,
            "duplicate_pdb_entity_rows": int(df.duplicated(["pdb_id", "entity_id"]).sum()) if {"pdb_id", "entity_id"}.issubset(df.columns) else "",
            "date_parse_failures_initial_release_date": sum(1 for x in df.get("initial_release_date", []) if nstr(x) and parse_date(x) is None),
            "missing_counts_json": json.dumps(miss, sort_keys=True, ensure_ascii=False),
            "primary_key_candidates": "pdb_id+entity_id; pdb_id+entity_id+auth_chains+asym_chains; full_row_hash",
        })
    write_csv(out_dir / "00_input_file_audit.csv", rows)
    counts = {
        "dataset_version": VERSION, "cutoff": CUTOFF.isoformat(),
        "expected_counts": {"all_evidence": 2782, "candidates": 47, "review": 1224, "all_d_like_excluded": 737, "false_positive_excluded": 774, "split_before": 28, "split_after": 19, "split_unknown": 0},
        "observed_counts": {k: int(len(frames[k])) for k in ["all_evidence", "candidates", "review", "all_d_like_excluded", "false_positive_excluded", "split_before", "split_after", "split_unknown"]},
    }
    counts["counts_reproduced"] = counts["expected_counts"] == counts["observed_counts"]
    c = frames["candidates"]
    counts["computed_candidate_split_counts"] = {
        "before_or_on_cutoff": int(sum(split(x) == "before_or_on_cutoff" for x in c["initial_release_date"])),
        "after_cutoff": int(sum(split(x) == "locked_post_cutoff_candidate" for x in c["initial_release_date"])),
        "release_date_unknown": int(sum(split(x) == "release_date_unknown" for x in c["initial_release_date"])),
    }
    (out_dir / "01_reproduced_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    return frames, {"hashes": hashes, "counts": counts, "audit_rows": rows}


def conflicts(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[dict]:
    seen = defaultdict(list)
    for k in ["candidates", "review", "all_d_like_excluded", "false_positive_excluded"]:
        for i, r in frames[k].iterrows():
            seen[key(r)].append({"source": FILES[k], "row_id": int(i) + 2, "verdict": r.get("local_d_verdict", ""), "confidence": r.get("confidence", "")})
    rows = []
    for k, vals in seen.items():
        if len({v["source"] for v in vals}) > 1 or len({v["verdict"] for v in vals}) > 1:
            p, e = k.split("|")
            rows.append({"conflict_type": "same_pdb_entity_multiple_classification_sources", "pdb_id_normalized": p, "entity_id": e, "source_files": ";".join(sorted({v["source"] for v in vals})), "original_labels": json.dumps(vals, ensure_ascii=False, sort_keys=True), "recommended_status": "manual_review", "evidence": "same pdb_id+entity_id appears in mutually exclusive source tables"})
    write_csv(out_dir / "02_classification_conflicts.csv", rows, ["conflict_type", "pdb_id_normalized", "entity_id", "source_files", "original_labels", "recommended_status", "evidence"])
    return rows


def release_audit(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[dict]:
    sets = {
        "before_or_on_cutoff": {key(r) for _, r in frames["split_before"].iterrows()},
        "locked_post_cutoff_candidate": {key(r) for _, r in frames["split_after"].iterrows()},
        "release_date_unknown": {key(r) for _, r in frames["split_unknown"].iterrows()},
    }
    rows = []
    for i, r in frames["candidates"].iterrows():
        k = key(r)
        existing = ";".join(name for name, vals in sets.items() if k in vals) or "missing_from_existing_split_files"
        d = parse_date(r.get("initial_release_date"))
        rows.append({"pdb_id_normalized": pdb(r.get("pdb_id")), "entity_id": nstr(r.get("entity_id")), "case_id_normalized": cid(r), "initial_release_date": r.get("initial_release_date", ""), "parsed_initial_release_date": d.isoformat() if d else "", "deposit_date": r.get("deposit_date", ""), "computed_split": split(r.get("initial_release_date")), "existing_split": existing, "split_match": existing == split(r.get("initial_release_date")), "date_field_used": "initial_release_date", "source_csv": FILES["candidates"], "source_row_id": int(i) + 2})
    write_csv(out_dir / "03_release_date_audit.csv", rows)
    return rows


def permanent_overlap(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[dict]:
    rows = []
    for k, df in frames.items():
        if k.startswith("split_"):
            continue
        for i, r in df.iterrows():
            p, c = pdb(r.get("pdb_id")), cid(r)
            if p in PERM_PDBS or c in PERM_CASES:
                rows.append({"source_csv": FILES[k], "source_row_id": int(i) + 2, "pdb_id_normalized": p, "entity_id": nstr(r.get("entity_id")), "case_id_normalized": c, "overlap_type": "pdb_id_or_case_id_overlap_with_permanent_postcutoff_test", "permanent_test_cases": ";".join(sorted(x for x in PERM_CASES if x.startswith(p))), "recommended_action": "exclude_from_development_and_rule_selection"})
    write_csv(out_dir / "05_permanent_test_overlap.csv", rows, ["source_csv", "source_row_id", "pdb_id_normalized", "entity_id", "case_id_normalized", "overlap_type", "permanent_test_cases", "recommended_action"])
    return rows


def residue_rows_for(row: pd.Series, cif: str, sp: str, src_row: int) -> tuple[list[dict], dict]:
    p, e, c = pdb(row["pdb_id"]), nstr(row["entity_id"]), cid(row)
    target_codes = all_target_codes(row)
    out = []
    for res in residues(atom_site(cif)):
        if str(res["entity"]) != e or res["comp"] not in target_codes:
            continue
        names = {a["label_atom_id"] for a in res["atoms"]} | {a["auth_atom_id"] for a in res["atoms"]}
        has_n, has_ca, has_c, has_o, has_cb = ("N" in names, "CA" in names, "C" in names, "O" in names, "CB" in names)
        side = sorted(names - {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3"})
        frame = has_n and has_ca and has_c
        supp = frame and bool(side)
        reason = "" if supp else ("MISSING_N_CA_C" if not frame else "NO_SIDECHAIN_ATOMS_FOR_CURRENT_MIRROR_RULE")
        out.append({
            "dataset_version": VERSION, "split": sp, "pdb_id": p, "case_id": c, "model_id": "1", "assembly_id": "not_materialized",
            "entity_id": e, "chain_id": res["auth_asym"] or res["label_asym"], "auth_seq_id": res["auth_seq"] or "", "label_seq_id": res["label_seq"] or "",
            "insertion_code": res["ins"], "residue_name": res["comp"], "ccd_id": res["comp"], "polymer_type": row.get("polymer_type", ""),
            "target_chirality": "D", "target_chirality_source": "polymer_entity_CCD_or_mmCIF_chemical_component_id_not_coordinate_chirality",
            "is_target_d_residue": True, "is_standard_amino_acid": False, "is_modified_amino_acid": True,
            "has_n": has_n, "has_ca": has_ca, "has_c": has_c, "has_o": has_o, "has_cb": has_cb, "has_sidechain_atoms": bool(side),
            "local_frame_available": frame, "correction_supported": supp, "correction_unsupported_reason": reason,
            "source_csv": FILES["candidates"], "source_row_id": src_row, "manual_review_required": not supp,
            "reference_observed_chirality": "not_computed_for_curation", "reference_geometry_valid": "not_computed_for_curation",
            "reference_allowed_for_curation": True, "reference_allowed_for_evaluation": True, "reference_allowed_for_correction": False,
            "notes": "target residue located by CCD/component identity inside candidate polymer entity",
        })
    return out, {"target_d_residue_count": len(out), "correction_supported_target_count": sum(bool(x["correction_supported"]) for x in out), "detected_codes": ";".join(sorted({x["ccd_id"] for x in out}))}


def build(input_dir: Path, out_dir: Path, allow_network: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, audit = audit_inputs(input_dir, out_dir)
    conf = conflicts(frames, out_dir)
    relrows = release_audit(frames, out_dir)
    overlaps = permanent_overlap(frames, out_dir)
    overlap_cases = {x["case_id_normalized"] for x in overlaps}
    conf_keys = {f"{x['pdb_id_normalized']}|{x['entity_id']}" for x in conf}
    cache = out_dir / "_cache_rcsb"
    case_rows, res_rows, by_case = [], [], {}
    for i, row in frames["candidates"].iterrows():
        p, e, c = pdb(row["pdb_id"]), nstr(row["entity_id"]), cid(row)
        sp = split(row.get("initial_release_date"))
        cif = fetch(f"https://files.rcsb.org/download/{p}.cif", cache / "mmcif" / f"{p}.cif", allow_network)
        ent_txt = fetch(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{p}/{e}", cache / "entity_json" / f"{p}_{e}.json", allow_network)
        ent_json = json.loads(ent_txt) if ent_txt else {}
        rr, meta = residue_rows_for(row, cif, sp, int(i) + 2) if cif else ([], {"target_d_residue_count": 0, "correction_supported_target_count": 0, "detected_codes": ""})
        res_rows.extend(rr)
        seq = canonical_sequence(ent_json) if ent_json else ""
        seq_sha = hashlib.sha256(seq.encode()).hexdigest() if seq else ""
        source_codes = ";".join(sorted(row_d_codes(row)))
        n_target = int(meta["target_d_residue_count"])
        n_support = int(meta["correction_supported_target_count"])
        text = f"{row.get('entry_title','')} {row.get('entity_description','')}"
        if c in overlap_cases or p in PERM_PDBS:
            status, reason = "excluded_leakage_risk", "PERMANENT_TEST_OVERLAP"
        elif f"{p}|{e}" in conf_keys:
            status, reason = "manual_review", "CONFLICTING_SOURCE_LABELS"
        elif ALL_D_TEXT.search(text):
            status, reason = "excluded_all_d_like", "ALL_D_LIKE"
        elif n_target == 0:
            status, reason = "manual_review", "INSUFFICIENT_EVIDENCE;TARGET_RESIDUE_NOT_IN_COORDINATES"
        elif n_support == 0:
            status, reason = "excluded_missing_target_coordinates", "MISSING_N_CA_C"
        else:
            status, reason = "eligible_local_d", "LOCAL_D_TARGET_RESIDUE_CONFIRMED_BY_CCD"
        allowed = status == "eligible_local_d" and sp == "before_or_on_cutoff"
        d = parse_date(row.get("initial_release_date"))
        case = {
            "dataset_version": VERSION, "source_csv": FILES["candidates"], "source_row_id": int(i) + 2,
            "pdb_id_raw": row.get("pdb_id", ""), "pdb_id_normalized": p, "entity_id": e, "case_id_normalized": c,
            "auth_chains_raw": row.get("auth_chains", ""), "asym_chains_raw": row.get("asym_chains", ""), "chain_id_normalized": row.get("auth_chains", "") or row.get("asym_chains", ""),
            "polymer_type": row.get("polymer_type", ""), "entity_description": row.get("entity_description", ""), "entry_title": row.get("entry_title", ""),
            "initial_release_date": row.get("initial_release_date", ""), "initial_release_date_parsed": d.isoformat() if d else "", "deposit_date": row.get("deposit_date", ""),
            "split": sp, "final_status": status, "reason_code": reason, "target_d_residue_count": n_target, "correction_supported_target_count": n_support,
            "sequence_length": len(seq), "sequence_one_letter_canonical": seq, "sequence_sha256": seq_sha, "d_ccd_codes_detected": meta["detected_codes"] or source_codes,
            "target_chirality_source": "CCD_component_id_in_candidate_polymer_entity; source text only used to select candidate rows",
            "reference_allowed_for_curation": True, "reference_allowed_for_evaluation": True, "reference_allowed_for_correction": False,
            "allowed_for_training": allowed, "locked_post_cutoff_candidate": sp == "locked_post_cutoff_candidate",
            "permanent_test_overlap": c in overlap_cases or p in PERM_PDBS, "duplicate_or_leakage_flag": False, "manual_review_required": status == "manual_review",
            "notes": "Full-complex AF3 input not materialized; complete complex-context materializer required before inference.",
        }
        case_rows.append(case)
        by_case[c] = case
    dup = duplicate_audit(case_rows, out_dir)
    cross = set()
    for d in dup:
        if str(d["cross_split_leakage_risk"]).lower() == "true":
            cross.update(pieces(d["case_ids"]))
    for r in case_rows:
        if r["case_id_normalized"] in cross:
            r["duplicate_or_leakage_flag"] = True
            if r["allowed_for_training"]:
                r["final_status"] = "excluded_leakage_risk"
                r["reason_code"] += ";DUPLICATE_SEQUENCE_CROSS_SPLIT"
                r["allowed_for_training"] = False
    write_csv(out_dir / "06_local_d_case_manifest.csv", case_rows, CASE_COLS)
    write_csv(out_dir / "07_local_d_residue_manifest.csv", res_rows, RES_COLS)
    write_csv(out_dir / "08_eligible_local_d_before_or_on_cutoff.csv", [r for r in case_rows if r["final_status"] == "eligible_local_d" and r["split"] == "before_or_on_cutoff"], CASE_COLS)
    write_csv(out_dir / "09_locked_local_d_after_cutoff.csv", [r for r in case_rows if r["split"] == "locked_post_cutoff_candidate"], CASE_COLS)
    write_csv(out_dir / "10_manual_review_queue.csv", [r for r in case_rows if r["manual_review_required"]], CASE_COLS)
    write_csv(out_dir / "11_excluded_cases.csv", [r for r in case_rows if str(r["final_status"]).startswith("excluded")], CASE_COLS)
    input_manifests(out_dir, case_rows, res_rows)
    mirror(out_dir, case_rows)
    return summary_report(out_dir, audit, conf, relrows, dup, overlaps, case_rows, res_rows)


def duplicate_audit(case_rows: list[dict], out_dir: Path) -> list[dict]:
    rows = []
    groups = [("same_pdb_id", "pdb_id_normalized"), ("same_pdb_entity", "case_id_normalized"), ("exact_sequence", "sequence_sha256")]
    for level, field in groups:
        g = defaultdict(list)
        for r in case_rows:
            if r.get(field):
                g[r[field]].append(r)
        for val, items in g.items():
            if len(items) > 1:
                splits = sorted({x["split"] for x in items})
                rows.append({"audit_level": level, "case_ids": ";".join(sorted({x["case_id_normalized"] for x in items})), "pdb_ids": ";".join(sorted({x["pdb_id_normalized"] for x in items})), "entity_ids": ";".join(sorted({x["entity_id"] for x in items})), "splits": ";".join(splits), "cross_split_leakage_risk": len(splits) > 1, "recommended_action": "exclude_or_lock_development_rows_from_rule_selection" if len(splits) > 1 else "retain_with_duplicate_note_or_manual_review", "evidence": f"{len(items)} rows share {field}={val}"})
    write_csv(out_dir / "04_duplicate_and_leakage_audit.csv", rows, ["audit_level", "case_ids", "pdb_ids", "entity_ids", "splits", "cross_split_leakage_risk", "recommended_action", "evidence"])
    return rows


def input_manifests(out_dir: Path, cases: list[dict], residues_: list[dict]) -> None:
    by_case = defaultdict(list)
    for r in residues_:
        by_case[r["case_id"]].append(r)
    seeds = random.Random(42).sample(range(1, 2**31 - 1), 30)
    std, at, corr, fut, seed_rows = [], [], [], [], []
    for i, seed in enumerate(seeds, 1):
        seed_rows.append({"seed_index": i, "inference_seed": seed, "master_seed": 42, "generator": "python.random.Random", "sampling_method": "sample_without_replacement"})
    for c in cases:
        ok = c["final_status"] == "eligible_local_d"
        base = {"dataset_version": VERSION, "case_id": c["case_id_normalized"], "pdb_id": c["pdb_id_normalized"], "entity_id": c["entity_id"], "split": c["split"], "sequence_one_letter_canonical": c["sequence_one_letter_canonical"], "sequence_sha256": c["sequence_sha256"], "input_json": "", "input_generation_status": "not_materialized_requires_full_complex_context_builder" if ok else "not_started_case_not_eligible_or_manual_review", "complex_context_preserved": False, "complex_context_note": "Current CSV gives candidate entity but not a complete AF3-ready complex graph.", "run_status": "not_started"}
        std.append({**base, "condition": "standard_af3", "D_CCD_enabled": False, "prompt_enabled": False, "notes": "No D chirality/CCD prompt is written to Standard AF3."})
        at.append({**base, "condition": "at_af3", "D_CCD_enabled": True, "prompt_enabled": True, "target_d_ccd_codes": ";".join(sorted({r["ccd_id"] for r in by_case[c["case_id_normalized"]]})), "notes": "AT-AF3 may alter only target D residue chemistry after full-complex materialization."})
        for r in by_case[c["case_id_normalized"]]:
            corr.append({"case_id": r["case_id"], "chain_id": r["chain_id"], "residue_id": str(r["auth_seq_id"]) + (str(r["insertion_code"]) if r["insertion_code"] else ""), "label_seq_id": r["label_seq_id"], "ccd_id": r["ccd_id"], "target_chirality": r["target_chirality"], "correction_supported": r["correction_supported"], "source_csv": r["source_csv"], "source_row_id": r["source_row_id"]})
        if ok:
            for si, seed in enumerate(seeds, 1):
                for cond in ("standard_af3", "at_af3"):
                    fut.append({"dataset_version": VERSION, "case_id": c["case_id_normalized"], "pdb_id": c["pdb_id_normalized"], "entity_id": c["entity_id"], "split": c["split"], "seed_index": si, "inference_seed": seed, "condition": cond, "run_status": "not_started", "input_json": "", "output_dir": f"runs_local_d_curated_v1/{cond}/{c['case_id_normalized']}/seed_{seed}", "same_seed_as_paired_condition": True, "old_regular_seed_used": seed in OLD_SEEDS, "notes": "future manifest only; no AF3 inference launched"})
    write_csv(out_dir / "12_standard_af3_input_manifest.csv", std)
    write_csv(out_dir / "13_at_af3_input_manifest.csv", at)
    write_csv(out_dir / "14_correction_target_manifest.csv", corr, ["case_id", "chain_id", "residue_id", "label_seq_id", "ccd_id", "target_chirality", "correction_supported", "source_csv", "source_row_id"])
    write_csv(out_dir / "15_future_master42_n30_generation_manifest.csv", fut)
    (out_dir / "inference_seeds_master42_n30.json").write_text(json.dumps({"master_seed": 42, "n_seeds": 30, "generator": "python.random.Random", "sampling_method": "sample_without_replacement", "sampling_range_start": 1, "sampling_range_stop_exclusive": 2147483647, "inference_seeds": seeds}, indent=2), encoding="utf-8")
    write_csv(out_dir / "inference_seeds_master42_n30.csv", seed_rows)


def mirror(out_dir: Path, cases: list[dict]) -> None:
    write_csv(out_dir / "16_mirror_af3_applicability.csv", [{"case_id": c["case_id_normalized"], "pdb_id": c["pdb_id_normalized"], "entity_id": c["entity_id"], "final_status": c["final_status"], "mirror_af3_applicable": False, "mirror_af3_invalid_reason": "Local-D/selective-D or multi-component context: whole-complex mirror would invert L partners or change interfaces; not a strict baseline."} for c in cases])


def summary_report(out_dir: Path, audit: dict, conf: list[dict], relrows: list[dict], dup: list[dict], overlaps: list[dict], cases: list[dict], res: list[dict]) -> dict:
    status, splits, reasons = Counter(r["final_status"] for r in cases), Counter(r["split"] for r in cases), Counter(r["reason_code"] for r in cases)
    supported = [r for r in res if str(r["correction_supported"]).lower() == "true"]
    candidate_perm = [r for r in cases if str(r.get("permanent_test_overlap", "")).lower() == "true"]
    summary = {"dataset_version": VERSION, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "input_source_hashes": audit["hashes"], "counts_reproduced": audit["counts"]["counts_reproduced"], "observed_counts": audit["counts"]["observed_counts"], "computed_candidate_split_counts": audit["counts"]["computed_candidate_split_counts"], "case_final_status_counts": dict(status), "candidate_split_counts": dict(splits), "reason_code_counts": dict(reasons), "n_cases": len(cases), "n_residue_manifest_rows": len(res), "n_correction_supported_target_residues": len(supported), "n_permanent_test_overlaps_all_source_csvs": len(overlaps), "n_candidate_manifest_permanent_test_overlaps": len(candidate_perm), "n_classification_conflicts": len(conf), "n_duplicate_or_leakage_audit_rows": len(dup), "reference_allowed_for_curation": True, "reference_allowed_for_evaluation": True, "reference_allowed_for_correction": False, "af3_inference_launched": False}
    (out_dir / "17_dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Local-D Curated v1 Audit Report", "", "This audit uses existing CSVs plus cached RCSB mmCIF/polymer_entity metadata. Target labels are CCD/component identity, not coordinate-derived chirality.", "", "## Input CSV schema"]
    lines += [f"- `{r['file_name']}` rows={r['n_rows']} cols={r['n_columns']} sha256={r['sha256']}" for r in audit["audit_rows"]]
    lines += ["", "## Reproduced counts", f"- counts reproduced: `{audit['counts']['counts_reproduced']}`", f"- observed: `{json.dumps(audit['counts']['observed_counts'], sort_keys=True)}`", f"- cutoff split: `{json.dumps(audit['counts']['computed_candidate_split_counts'], sort_keys=True)}`", "", "## Final case classification counts", "| status | count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in status.most_common()]
    lines += ["", "## Cutoff counts", "| split | count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in splits.most_common()]
    lines += ["", "## Leakage / duplicates", f"- permanent test overlap rows across all source CSVs: {len(overlaps)}", f"- permanent test overlap rows in 47 candidate manifest: {len(candidate_perm)}", f"- classification conflicts: {len(conf)}", f"- duplicate/leakage rows: {len(dup)}", "- MMseqs2/CD-HIT clustering was not run; the exact-sequence duplicate audit is complete and sequence clustering is outside this audit.", "", "## Residue-level targets", f"- target D-residue rows: {len(res)}", f"- deterministic correction-supported target rows: {len(supported)}", "", "## AF3 input status", "No AF3 inference was launched. Full-complex AF3 JSONs are not materialized because source CSVs identify candidate entities, not complete AF3-ready complex graphs.", "", "## Reporting language", "Use `target D-residue proportion`, not `D-chirality accuracy`, at this curation stage.", "", "## Commands", "```bash", "python scripts/local_d_curated_v1/audit_local_d_inputs.py --input-dir data/local_d --output-dir data/local_d/local_d_curated_v1", "python scripts/local_d_curated_v1/build_local_d_manifests.py --input-dir data/local_d --output-dir data/local_d/local_d_curated_v1 --allow-network", "python scripts/local_d_curated_v1/check_local_d_leakage.py --output-dir data/local_d/local_d_curated_v1", "python scripts/local_d_curated_v1/generate_local_d_af3_inputs.py --output-dir data/local_d/local_d_curated_v1", "python scripts/local_d_curated_v1/validate_local_d_curated_v1.py --output-dir data/local_d/local_d_curated_v1", "```"]
    (out_dir / "18_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate(out_dir: Path) -> dict:
    required = [f"{i:02d}_{name}" for i, name in [
        (0, "input_file_audit.csv"), (1, "reproduced_counts.json"), (2, "classification_conflicts.csv"), (3, "release_date_audit.csv"), (4, "duplicate_and_leakage_audit.csv"), (5, "permanent_test_overlap.csv"), (6, "local_d_case_manifest.csv"), (7, "local_d_residue_manifest.csv"), (8, "eligible_local_d_before_or_on_cutoff.csv"), (9, "locked_local_d_after_cutoff.csv"), (10, "manual_review_queue.csv"), (11, "excluded_cases.csv"), (12, "standard_af3_input_manifest.csv"), (13, "at_af3_input_manifest.csv"), (14, "correction_target_manifest.csv"), (15, "future_master42_n30_generation_manifest.csv"), (16, "mirror_af3_applicability.csv"), (17, "dataset_summary.json"), (18, "audit_report.md")]]
    missing = [x for x in required if not (out_dir / x).exists()]
    errors = []
    if missing:
        errors.append("missing_outputs:" + ";".join(missing))
    case = pd.read_csv(out_dir / "06_local_d_case_manifest.csv", dtype=str, keep_default_na=False) if not missing else pd.DataFrame()
    res = pd.read_csv(out_dir / "07_local_d_residue_manifest.csv", dtype=str, keep_default_na=False) if not missing else pd.DataFrame()
    corr = pd.read_csv(out_dir / "14_correction_target_manifest.csv", dtype=str, keep_default_na=False) if not missing else pd.DataFrame()
    fut = pd.read_csv(out_dir / "15_future_master42_n30_generation_manifest.csv", dtype=str, keep_default_na=False) if not missing else pd.DataFrame()
    if not case.empty:
        for c in CASE_COLS:
            if c not in case.columns:
                errors.append("missing_case_column:" + c)
        if case["case_id_normalized"].eq("").any() or case["split"].eq("").any() or case["final_status"].eq("").any():
            errors.append("empty_case_id_split_or_status")
        if len(case[(case["allowed_for_training"].str.lower() == "true") & (case["split"] != "before_or_on_cutoff")]):
            errors.append("postcutoff_allowed_for_training")
        if len(case[(case["pdb_id_normalized"].isin(PERM_PDBS)) & (case["allowed_for_training"].str.lower() == "true")]):
            errors.append("permanent_test_allowed_for_training")
    if not res.empty and not set(res["case_id"]).issubset(set(case["case_id_normalized"])):
        errors.append("residue_rows_without_case")
    bad_cols = [c for c in corr.columns if any(t in c.lower() for t in ["reference", "rmsd", "tm_score", "observed_chirality"])]
    if bad_cols:
        errors.append("correction_target_manifest_has_reference_like_columns:" + ";".join(bad_cols))
    if not fut.empty:
        seeds = sorted({int(x) for x in fut["inference_seed"] if nstr(x)})
        if any(s in OLD_SEEDS for s in seeds):
            errors.append("old_regular_seed_present")
        if seeds and seeds != sorted(random.Random(42).sample(range(1, 2**31 - 1), 30)):
            errors.append("future_seed_set_not_master42_n30")
    result = {"validation_passed": not errors, "errors": errors, "n_cases": int(len(case)), "n_residue_rows": int(len(res))}
    (out_dir / "validation_local_d_curated_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise SystemExit("Local-D curated v1 validation failed: " + "; ".join(errors))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["audit", "build", "leakage", "inputs", "validate", "all"])
    p.add_argument("--input-dir", default=str(LOCAL_D_ROOT))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--allow-network", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "audit":
        frames, audit = audit_inputs(Path(a.input_dir), out)
        conf = conflicts(frames, out)
        relrows = release_audit(frames, out)
        overlaps = permanent_overlap(frames, out)
        result = {"counts": audit["counts"], "classification_conflicts": len(conf), "release_rows": len(relrows), "permanent_overlap_rows": len(overlaps)}
    elif a.command in {"build", "all"}:
        result = build(Path(a.input_dir), out, a.allow_network or a.command == "all")
        validate(out)
    elif a.command == "leakage":
        case = pd.read_csv(out / "06_local_d_case_manifest.csv", dtype=str, keep_default_na=False).to_dict("records")
        result = {"duplicate_or_leakage_rows": len(duplicate_audit(case, out))}
    elif a.command == "inputs":
        case = pd.read_csv(out / "06_local_d_case_manifest.csv", dtype=str, keep_default_na=False).to_dict("records")
        res = pd.read_csv(out / "07_local_d_residue_manifest.csv", dtype=str, keep_default_na=False).to_dict("records")
        input_manifests(out, case, res)
        result = {"input_manifests_rewritten": True}
    else:
        result = validate(out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
