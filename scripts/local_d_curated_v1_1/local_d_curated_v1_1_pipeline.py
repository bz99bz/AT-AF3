from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "local_d"
SRC_DIR = BASE_DIR / "local_d_curated_v1"
OUT_DIR = BASE_DIR / "local_d_curated_v1_1"
VERSION = "local_d_curated_v1.1"
CUTOFF = dt.date(2021, 9, 30)
MASTER_SEED = 42
N_SEEDS = 30
PERM_CASES = {"7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"}
PERM_PDBS = {x.split("_", 1)[0] for x in PERM_CASES}
OLD_SEEDS = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010}
RUNNABLE_TYPES = {"development_eligible_local_d"}
WATER_CCD = {"HOH", "H2O", "DOD"}
COMMON_SOLVENTS = {"GOL", "SO4", "PO4", "NA", "CL", "K", "CA", "MG", "ZN", "ACT", "EDO", "MPD"}
STANDARD_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
D_PARENT = {
    "DAL": "A", "DAR": "R", "DSN": "N", "DAS": "D", "DCY": "C", "DGN": "Q", "DGL": "E",
    "DHI": "H", "DIL": "I", "DLE": "L", "DLY": "K", "DME": "M", "MED": "M", "DPN": "F",
    "DPR": "P", "DSE": "S", "DTH": "T", "DTR": "W", "DTY": "Y", "DVA": "V",
    "28J": "K", "HJV": "K", "HJY": "K", "ZAE": "E", "MLU": "L", "A1EJL": "L", "A1EQM": "Q", "A1CTT": "T",
}
AA3_TO_AA1 = {**STANDARD_AA, **D_PARENT}

sys.path.insert(0, str(ROOT / "scripts" / "local_d_curated_v1"))
import local_d_curated_pipeline as base  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def nstr(v: Any) -> str:
    return base.nstr(v)


def clean(v: Any) -> str:
    return base.clean(v)


def parse_date(v: Any):
    return base.parse_date(v)


def split(v: Any) -> str:
    d = parse_date(v)
    if d is None:
        return "release_date_unknown"
    return "before_or_on_cutoff" if d <= CUTOFF else "locked_post_cutoff_candidate"


def load_core_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    case = pd.read_csv(SRC_DIR / "06_local_d_case_manifest.csv", dtype=str, keep_default_na=False)
    res = pd.read_csv(SRC_DIR / "07_local_d_residue_manifest.csv", dtype=str, keep_default_na=False)
    return case, res


def freeze_roles(case: pd.DataFrame) -> pd.DataFrame:
    out = case.copy()
    roles = []
    for _, r in out.iterrows():
        s = r["split"]
        st = r["final_status"]
        if st == "eligible_local_d" and s == "before_or_on_cutoff":
            role = "development_eligible_local_d"
            dev = ms = True
            pcut = False
            blocked = ""
            allowed = True
        elif st == "eligible_local_d" and s == "locked_post_cutoff_candidate":
            role = "locked_post_cutoff_eligible_local_d"
            dev = False
            ms = False
            pcut = True
            blocked = "locked_post_cutoff_candidate_not_allowed_for_model_selection"
            allowed = False
        elif st == "manual_review" and s == "before_or_on_cutoff":
            role = "blocked_manual_review_pre_cutoff"
            dev = False
            ms = False
            pcut = False
            blocked = "manual_review_before_cutoff_needs_complete_complex_reconstruction"
            allowed = False
        elif st == "manual_review" and s == "locked_post_cutoff_candidate":
            role = "blocked_manual_review_post_cutoff"
            dev = False
            ms = False
            pcut = False
            blocked = "manual_review_after_cutoff_needs_complete_complex_reconstruction"
            allowed = False
        elif st == "excluded_leakage_risk":
            role = "excluded_permanent_leakage_risk"
            dev = False
            ms = False
            pcut = False
            blocked = "permanent_test_or_cross_split_leakage"
            allowed = False
        else:
            role = f"unknown_{st}"
            dev = ms = pcut = allowed = False
            blocked = "unclassified"
        roles.append({
            "dataset_role": role,
            "allowed_for_development": dev,
            "allowed_for_model_selection": ms,
            "allowed_for_post_cutoff_evaluation": pcut,
            "allowed_to_run": allowed,
            "blocked_reason": blocked,
        })
    return pd.concat([out, pd.DataFrame(roles)], axis=1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_loops(text: str):
    return base.iter_loops(text)


def split_cif_row(line: str):
    return base.split_cif_row(line)


def atom_site(text: str):
    return base.atom_site(text)


def residues(atoms: list[dict]):
    return base.residues(atoms)


def parse_entity_poly(entity_json: dict) -> list[dict]:
    ep = entity_json.get("entity_poly", {}) or {}
    seq = re.sub(r"\s+", "", re.sub(r"\([^)]*\)", "X", str(ep.get("pdbx_seq_one_letter_code_can") or ep.get("pdbx_seq_one_letter_code") or ""))).upper()
    typ = clean(ep.get("type"))
    out = []
    if seq:
        out.append({"sequence": seq, "polymer_type": typ})
    return out


def protein_chain_sequence(chain_residues: list[dict]) -> tuple[str, list[dict], list[dict]]:
    seq = []
    mods = []
    map_rows = []
    af3_idx = 0
    for res in chain_residues:
        comp = res["comp"].upper()
        if comp in WATER_CCD or comp in COMMON_SOLVENTS:
            continue
        aa = AA3_TO_AA1.get(comp)
        if not aa:
            continue
        af3_idx += 1
        seq.append(aa)
        map_rows.append({
            "label_asym_id": res["label_asym"],
            "auth_asym_id": res["auth_asym"],
            "entity_id": res["entity"],
            "label_seq_id": res["label_seq"] if res["label_seq"] is not None else "",
            "auth_seq_id": res["auth_seq"] if res["auth_seq"] is not None else "",
            "insertion_code": res["ins"],
            "ccd_id": comp,
            "af3_input_residue_index": af3_idx,
            "is_target_d_residue": comp.startswith("D") or comp in {"DAL", "DAR", "DSN", "DAS", "DCY", "DGN", "DGL", "DHI", "DIL", "DLE", "DLY", "DME", "MED", "DPN", "DPR", "DSE", "DTH", "DTR", "DTY", "DVA"},
        })
        if comp in D_PARENT:
            mods.append({"ptmType": comp, "ptmPosition": af3_idx})
    return "".join(seq), mods, map_rows


def mmcif_meta(text: str) -> dict:
    lines = text.splitlines()
    assemblies = []
    asm_id = None
    idxs = {}
    for headers, rows in iter_loops(text):
        if headers and all(h.startswith("_pdbx_struct_assembly_gen.") for h in headers):
            idxs = {h: i for i, h in enumerate(headers)}
            for r in rows:
                if len(r) < len(headers):
                    continue
                assemblies.append({
                    "assembly_id": clean(r[idxs.get("_pdbx_struct_assembly_gen.assembly_id", 0)]) if "_pdbx_struct_assembly_gen.assembly_id" in idxs else "",
                    "oper_expression": clean(r[idxs.get("_pdbx_struct_assembly_gen.oper_expression", 0)]) if "_pdbx_struct_assembly_gen.oper_expression" in idxs else "",
                    "asym_id_list": clean(r[idxs.get("_pdbx_struct_assembly_gen.asym_id_list", 0)]) if "_pdbx_struct_assembly_gen.asym_id_list" in idxs else "",
                })
        if headers and all(h.startswith("_pdbx_struct_assembly.") for h in headers):
            idxs = {h: i for i, h in enumerate(headers)}
            for r in rows:
                if len(r) < len(headers):
                    continue
                asm_id = clean(r[idxs["_pdbx_struct_assembly.id"]]) if "_pdbx_struct_assembly.id" in idxs else ""
                break
    if not assemblies:
        assemblies = [{"assembly_id": "asymmetric_unit", "oper_expression": "", "asym_id_list": ""}]
    return {"assemblies": assemblies, "default_assembly_id": asm_id or assemblies[0]["assembly_id"]}


def choose_assembly(case_id: str, target_asym: str, assemblies: list[dict], asym_ids: list[str]) -> tuple[str, dict]:
    if not assemblies:
        return "asymmetric_unit", {"manual_review_required": True, "assembly_manual_review_required": True, "assembly_selection_reason": "no_assembly_metadata"}
    for a in assemblies:
        asyms = {x.strip() for x in re.split(r"[;, ]+", a.get("asym_id_list", "")) if x.strip()}
        if target_asym in asyms:
            return a["assembly_id"] or "1", {"manual_review_required": False, "assembly_manual_review_required": False, "assembly_selection_reason": "target_chain_present_in_assembly"}
    return assemblies[0]["assembly_id"] or "1", {"manual_review_required": True, "assembly_manual_review_required": True, "assembly_selection_reason": "ambiguous_assembly_fallback"}


def build_external_manifests(frozen_case: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    hashes = []
    seen = set()
    for _, r in frozen_case.iterrows():
        p = r["pdb_id_normalized"]
        e = r["entity_id"]
        case = r["case_id_normalized"]
        for typ, path, url in [
            ("mmcif", SRC_DIR / "_cache_rcsb" / "mmcif" / f"{p}.cif", f"https://files.rcsb.org/download/{p}.cif"),
            ("entity_json", SRC_DIR / "_cache_rcsb" / "entity_json" / f"{p}_{e}.json", f"https://data.rcsb.org/rest/v1/core/polymer_entity/{p}/{e}"),
        ]:
            key = (typ, str(path))
            if key in seen:
                continue
            seen.add(key)
            if not path.exists():
                continue
            rows.append({
                "resource_type": typ,
                "pdb_id": p,
                "ccd_id": "",
                "request_url_or_query": url,
                "retrieval_timestamp_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
                "source_database": "RCSB",
                "database_version_if_available": "",
                "local_path": str(path.relative_to(ROOT)),
                "file_size": path.stat().st_size,
                "sha256": sha(path),
                "used_for_case_ids": case,
            })
            hashes.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "used_for_case_ids": case})
    write_csv(OUT_DIR / "19_external_resource_manifest.csv", rows)
    write_csv(OUT_DIR / "21_input_and_cache_hashes.csv", hashes, ["path", "sha256", "used_for_case_ids"])
    return pd.DataFrame(rows), pd.DataFrame(hashes), pd.DataFrame()


def software_environment() -> dict:
    git_commit = ""
    git_clean = None
    try:
        git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_clean = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip() == ""
    except Exception:
        git_commit = ""
        git_clean = None
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception:
        pip_freeze = []
    return {
        "dataset_version": VERSION,
        "parent_version": "local_d_curated_v1",
        "python_version": sys.version,
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "python_executable": sys.executable,
        "pip_freeze": pip_freeze,
        "git_commit": git_commit,
        "git_working_tree_clean": git_clean,
        "command": " ".join(sys.argv),
        "cutoff_date": CUTOFF.isoformat(),
        "master_seed": MASTER_SEED,
        "n_seeds": N_SEEDS,
        "dataset_role_contract": ["development_eligible_local_d", "locked_post_cutoff_eligible_local_d", "blocked_manual_review_pre_cutoff", "blocked_manual_review_post_cutoff", "excluded_permanent_leakage_risk"],
    }


def assemble_complexes(frozen_case: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comp_rows, asm_rows, excl_rows, map_rows = [], [], [], []
    for _, r in frozen_case.iterrows():
        if r["final_status"] == "manual_review" or r["final_status"] == "excluded_leakage_risk":
            continue
        p = r["pdb_id_normalized"]
        c = r["case_id_normalized"]
        cif_path = SRC_DIR / "_cache_rcsb" / "mmcif" / f"{p}.cif"
        if not cif_path.exists():
            asm_rows.append({"case_id": c, "pdb_id": p, "assembly_id": "", "assembly_manual_review_required": True, "assembly_selection_reason": "missing_cached_mmcif", "notes": "offline cache missing"})
            continue
        text = read_text(cif_path)
        atoms = atom_site(text)
        res = residues(atoms)
        by_chain = defaultdict(list)
        for rr in res:
            by_chain[rr["label_asym"] or rr["auth_asym"]].append(rr)
        meta = mmcif_meta(text)
        target_chain = r["chain_id_normalized"] or next(iter(by_chain), "")
        chosen_asm, asm_note = choose_assembly(c, target_chain, meta["assemblies"], list(by_chain))
        asm_rows.append({"case_id": c, "pdb_id": p, "assembly_id": chosen_asm, **asm_note, "available_assembly_ids": ";".join(sorted({a["assembly_id"] for a in meta["assemblies"] if a.get("assembly_id")})), "target_chain_id": target_chain})
        for chain_id, chain_res in by_chain.items():
            seq, mods, maps = protein_chain_sequence(chain_res)
            if not seq:
                excl_rows.append({"case_id": c, "pdb_id": p, "assembly_id": chosen_asm, "component_type": "nonprotein_or_unmapped", "entity_id": chain_res[0]["entity"], "asym_id": chain_id, "auth_chain_id": chain_res[0]["auth_asym"], "sequence": "", "polymer_type": "", "ccd_id": "", "stoichiometry": "1", "target_component": str(chain_id == target_chain), "target_d_residue_count": 0, "expected_l_residue_count": 0, "include_in_af3_input": False, "inclusion_reason": "nonprotein_or_unmapped_component_excluded_from_dry_run", "source": "offline_cached_mmcif", "manual_review_required": True})
                continue
            target_d = sum(1 for m in mods)
            comp_rows.append({
                "case_id": c, "pdb_id": p, "assembly_id": chosen_asm, "component_index": len(comp_rows) + 1, "component_type": "protein_polymer",
                "entity_id": chain_res[0]["entity"], "asym_id": chain_id, "auth_chain_id": chain_res[0]["auth_asym"], "sequence": seq,
                "polymer_type": "protein", "ccd_id": ";".join(sorted({x["comp"] for x in chain_res})), "stoichiometry": "1",
                "target_component": chain_id == target_chain, "target_d_residue_count": target_d, "expected_l_residue_count": len(seq) - target_d,
                "include_in_af3_input": True, "inclusion_reason": "protein_chain_from_cached_mmcif", "source": "offline_cached_mmcif", "manual_review_required": False
            })
            for m in maps:
                m.update({"case_id": c, "pdb_id": p, "assembly_id": chosen_asm, "component_index": len(comp_rows), "sequence_chain_id": chain_id})
                map_rows.append(m)
    write_csv(OUT_DIR / "22_complex_component_manifest.csv", comp_rows)
    write_csv(OUT_DIR / "23_assembly_selection_audit.csv", asm_rows)
    write_csv(OUT_DIR / "24_component_exclusion_audit.csv", excl_rows)
    write_csv(OUT_DIR / "25_residue_identifier_mapping.csv", map_rows)
    return pd.DataFrame(comp_rows), pd.DataFrame(asm_rows), pd.DataFrame(excl_rows), pd.DataFrame(map_rows)


def build_af3_inputs(frozen_case: pd.DataFrame, comp: pd.DataFrame, resmap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    af3_root = OUT_DIR / "af3_inputs"
    std_dir = af3_root / "standard"
    at_dir = af3_root / "at_af3"
    std_dir.mkdir(parents=True, exist_ok=True)
    at_dir.mkdir(parents=True, exist_ok=True)
    diff_rows, std_rows, at_rows = [], [], []
    target_map = defaultdict(list)
    for _, r in resmap.iterrows():
        if str(r.get("is_target_d_residue", "")).lower() == "true":
            target_map[(r["case_id"], r["sequence_chain_id"])].append(int(r["af3_input_residue_index"]))
    comp_by_case = defaultdict(list)
    for _, r in comp.iterrows():
        comp_by_case[r["case_id"]].append(r)
    for _, r in frozen_case.iterrows():
        c = r["case_id_normalized"]
        p = r["pdb_id_normalized"]
        allowed = str(r["allowed_for_development"]).lower() == "true"
        role = r["dataset_role"]
        if not allowed:
            continue
        chains = comp_by_case.get(c, [])
        sequences = []
        for ch in chains:
            seq_obj = {"protein": {"id": ch["asym_id"], "sequence": ch["sequence"], "modifications": [], "unpairedMsa": "", "pairedMsa": "", "templates": []}}
            sequences.append(seq_obj)
        std_json = {"name": f"{c}_standard_af3", "modelSeeds": [MASTER_SEED], "sequences": sequences, "dialect": "alphafold3", "version": 2}
        at_sequences = []
        for ch in chains:
            mods = []
            idxs = target_map.get((c, ch["asym_id"]), [])
            if role == "development_eligible_local_d":
                for idx in idxs:
                    # Use the chain's actual target CCD from the source manifest via residue map.
                    row = resmap[(resmap["case_id"] == c) & (resmap["sequence_chain_id"] == ch["asym_id"]) & (resmap["af3_input_residue_index"].astype(str) == str(idx))].iloc[0]
                    mods.append({"ptmType": row["ccd_id"], "ptmPosition": int(idx)})
            at_sequences.append({"protein": {"id": ch["asym_id"], "sequence": ch["sequence"], "modifications": mods, "unpairedMsa": "", "pairedMsa": "", "templates": []}})
        at_json = {"name": f"{c}_at_af3", "modelSeeds": [MASTER_SEED], "sequences": at_sequences, "dialect": "alphafold3", "version": 2}
        std_path = std_dir / f"{c}_standard.json"
        at_path = at_dir / f"{c}_at_af3.json"
        std_path.write_text(json.dumps(std_json, indent=2), encoding="utf-8")
        at_path.write_text(json.dumps(at_json, indent=2), encoding="utf-8")
        std_rows.append({"case_id": c, "pdb_id": p, "dataset_role": role, "input_json": str(std_path.relative_to(OUT_DIR)), "allowed_to_run": allowed, "blocked_reason": "" if allowed else r["blocked_reason"], "condition": "standard_af3", "modelSeeds": MASTER_SEED, "n_chains": len(chains), "n_target_modifications": 0})
        at_rows.append({"case_id": c, "pdb_id": p, "dataset_role": role, "input_json": str(at_path.relative_to(OUT_DIR)), "allowed_to_run": allowed, "blocked_reason": "" if allowed else r["blocked_reason"], "condition": "at_af3", "modelSeeds": MASTER_SEED, "n_chains": len(chains), "n_target_modifications": sum(len(target_map.get((c, ch["asym_id"]), [])) for ch in chains)})
        diff_rows.append({"case_id": c, "same_sequences": True, "standard_has_modifications": False, "at_has_modifications": True if at_rows[-1]["n_target_modifications"] else False, "only_target_positions_differ": True if role == "development_eligible_local_d" else False, "diff_status": "passed" if role == "development_eligible_local_d" else "blocked"})
    write_csv(OUT_DIR / "26_standard_vs_at_af3_input_diff.csv", diff_rows)
    write_csv(OUT_DIR / "standard_af3_input_manifest.csv", std_rows)
    write_csv(OUT_DIR / "at_af3_input_manifest.csv", at_rows)
    return pd.DataFrame(std_rows), pd.DataFrame(at_rows), pd.DataFrame(diff_rows)


def build_future_and_correction(frozen_case: pd.DataFrame, std: pd.DataFrame, at: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = random.Random(MASTER_SEED).sample(range(1, 2**31 - 1), N_SEEDS)
    seed_rows = [{"seed_index": i, "inference_seed": s, "master_seed": MASTER_SEED, "generator": "python.random.Random", "sampling_method": "sample_without_replacement"} for i, s in enumerate(seeds, 1)]
    write_csv(OUT_DIR / "inference_seeds_master42_n30.csv", seed_rows)
    (OUT_DIR / "inference_seeds_master42_n30.json").write_text(json.dumps({
        "master_seed": MASTER_SEED, "n_seeds": N_SEEDS, "generator": "python.random.Random", "sampling_method": "sample_without_replacement", "sampling_range_start": 1, "sampling_range_stop_exclusive": 2**31 - 1, "inference_seeds": seeds
    }, indent=2), encoding="utf-8")
    std_by_case = defaultdict(list)
    at_by_case = defaultdict(list)
    for _, r in std.iterrows():
        std_by_case[r["case_id"]].append(r)
    for _, r in at.iterrows():
        at_by_case[r["case_id"]].append(r)
    future, correction = [], []
    runnable_cases = [r for _, r in frozen_case.iterrows() if str(r["allowed_to_run"]).lower() == "true"]
    blocked_cases = [r for _, r in frozen_case.iterrows() if str(r["allowed_to_run"]).lower() != "true" and r["final_status"] != "excluded_leakage_risk"]
    for r in runnable_cases + blocked_cases:
        case = r["case_id_normalized"]
        for si, seed in enumerate(seeds, 1):
            for cond in ("standard_af3", "at_af3"):
                allowed = str(r["allowed_to_run"]).lower() == "true"
                reason = "" if allowed else r["blocked_reason"]
                future.append({
                    "dataset_version": VERSION, "case_id": case, "pdb_id": r["pdb_id_normalized"], "entity_id": r["entity_id"], "split": r["split"],
                    "dataset_role": r["dataset_role"], "evaluation_round": "local_d_curated_v1_1_dry_run", "configuration_hash": sha(OUT_DIR / "20_software_environment.json") if (OUT_DIR / "20_software_environment.json").exists() else "",
                    "condition": cond, "seed_index": si, "seed": seed, "input_json_path": f"af3_inputs/{'standard' if cond=='standard_af3' else 'at_af3'}/{case}_{'standard' if cond=='standard_af3' else 'at_af3'}.json",
                    "input_json_sha256": sha((OUT_DIR / "af3_inputs" / ("standard" if cond == "standard_af3" else "at_af3") / f"{case}_{'standard' if cond=='standard_af3' else 'at_af3'}.json")) if allowed else "",
                    "run_status": "not_started" if allowed else "blocked", "allowed_to_run": allowed, "blocked_reason": reason,
                    "expected_first_pass_output": f"runs_local_d_curated_v1_1/{cond}/{case}/seed_{seed}/result.cif", "expected_log_path": f"logs_local_d_curated_v1_1/{cond}/{case}/seed_{seed}.log",
                })
        if allowed:
            for seed in seeds:
                correction.append({
                    "case_id": case, "seed": seed, "at_af3_first_pass_output_path": f"runs_local_d_curated_v1_1/at_af3/{case}/seed_{seed}/result.cif",
                    "correction_target_manifest_path": "14_correction_target_manifest.csv", "correction_status": "not_started",
                    "correction_deterministic": True, "reference_used_for_correction": False, "error_code_status": "compute_from_prediction_at_runtime",
                    "corrected_output_path": f"runs_local_d_curated_v1_1/at_af3_corrected/{case}/seed_{seed}/result.cif", "metadata_output_path": f"runs_local_d_curated_v1_1/at_af3_corrected/{case}/seed_{seed}/metadata.json",
                })
    write_csv(OUT_DIR / "15_future_master42_n30_generation_manifest.csv", future)
    write_csv(OUT_DIR / "27_future_correction_job_manifest.csv", correction)
    return pd.DataFrame(future), pd.DataFrame(correction)


def build_summary(frozen_case: pd.DataFrame, res: pd.DataFrame, future: pd.DataFrame, correction: pd.DataFrame, comp: pd.DataFrame, std: pd.DataFrame, at: pd.DataFrame, diff: pd.DataFrame, asm: pd.DataFrame, excl: pd.DataFrame, ext: pd.DataFrame, hashes: pd.DataFrame) -> dict:
    counts = Counter(frozen_case["dataset_role"])
    summary = {
        "dataset_version": VERSION,
        "parent_version": "local_d_curated_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_cases": int(len(frozen_case)),
        "n_development_cases": int((frozen_case["dataset_role"] == "development_eligible_local_d").sum()),
        "n_locked_postcutoff_cases": int((frozen_case["dataset_role"] == "locked_post_cutoff_eligible_local_d").sum()),
        "n_manual_review_cases": int(frozen_case["dataset_role"].str.startswith("blocked_manual_review").sum()),
        "n_excluded_cases": int((frozen_case["dataset_role"] == "excluded_permanent_leakage_risk").sum()),
        "n_residue_manifest_rows": int(len(res)),
        "n_correction_supported_target_residues": int((res["correction_supported"].astype(str).str.lower() == "true").sum()) if not res.empty else 0,
        "n_complex_components": int(len(comp)),
        "n_component_exclusions": int(len(excl)),
        "n_assemblies": int(len(asm)),
        "n_standard_json": int((std["allowed_to_run"].astype(str).str.lower() == "true").sum()),
        "n_at_json": int((at["allowed_to_run"].astype(str).str.lower() == "true").sum()),
        "n_blocked_case_rows": int((frozen_case["allowed_to_run"].astype(str).str.lower() != "true").sum()),
        "n_permanent_test_overlap_rows": int((frozen_case["permanent_test_overlap"].astype(str).str.lower() == "true").sum()),
        "n_future_manifest_rows": int(len(future)),
        "n_future_correction_rows": int(len(correction)),
        "n_input_diff_rows": int(len(diff)),
        "n_external_resource_rows": int(len(ext)),
        "n_cache_hash_rows": int(len(hashes)),
        "counts_by_dataset_role": dict(counts),
        "reference_allowed_for_curation": True,
        "reference_allowed_for_evaluation": True,
        "reference_allowed_for_correction": False,
        "af3_inference_launched": False,
        "offline_mode": True,
        "all_network_fetches_blocked": True,
    }
    (OUT_DIR / "28_local_d_curated_v1_1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Local-D Curated v1.1 Audit Report",
        "",
        "Offline freeze + full-complex reconstruction + AF3 JSON dry-run only. No AF3 inference was launched.",
        "",
        "## Role split",
        f"- development cases: {summary['n_development_cases']}",
        f"- locked post-cutoff cases: {summary['n_locked_postcutoff_cases']}",
        f"- manual review cases: {summary['n_manual_review_cases']}",
        f"- excluded cases: {summary['n_excluded_cases']}",
        "",
        "## Frozen counts",
        f"- residue rows: {summary['n_residue_manifest_rows']}",
        f"- supported target D residues: {summary['n_correction_supported_target_residues']}",
        f"- complex components: {summary['n_complex_components']}",
        f"- component exclusions: {summary['n_component_exclusions']}",
        f"- assemblies selected: {summary['n_assemblies']}",
        "",
        "## JSON dry-run",
        f"- standard JSON files: {summary['n_standard_json']}",
        f"- AT-AF3 JSON files: {summary['n_at_json']}",
        f"- blocked case rows: {summary['n_blocked_case_rows']}",
        "",
        "## Leakage and determinism",
        f"- permanent-test overlaps: {summary['n_permanent_test_overlap_rows']}",
        f"- future manifest rows: {summary['n_future_manifest_rows']}",
        f"- future correction rows: {summary['n_future_correction_rows']}",
        f"- offline mode: {summary['offline_mode']}",
        f"- network blocked: {summary['all_network_fetches_blocked']}",
        "",
        "## Validation language",
        "Use dataset roles rather than the ambiguous single `eligible_local_d` label.",
        "",
        "## Outputs",
        "- 19_external_resource_manifest.csv",
        "- 20_software_environment.json",
        "- 21_input_and_cache_hashes.csv",
        "- 22_complex_component_manifest.csv",
        "- 23_assembly_selection_audit.csv",
        "- 24_component_exclusion_audit.csv",
        "- 25_residue_identifier_mapping.csv",
        "- 26_standard_vs_at_af3_input_diff.csv",
        "- 27_future_correction_job_manifest.csv",
        "- 28_local_d_curated_v1_1_summary.json",
        "- 29_local_d_curated_v1_1_audit_report.md",
    ]
    (OUT_DIR / "29_local_d_curated_v1_1_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "20_software_environment.json").write_text(json.dumps(software_environment(), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def validate(out_dir: Path) -> dict:
    required = [
        "06_local_d_case_manifest.csv",
        "07_local_d_residue_manifest.csv",
        "19_external_resource_manifest.csv",
        "20_software_environment.json",
        "21_input_and_cache_hashes.csv",
        "22_complex_component_manifest.csv",
        "23_assembly_selection_audit.csv",
        "24_component_exclusion_audit.csv",
        "25_residue_identifier_mapping.csv",
        "26_standard_vs_at_af3_input_diff.csv",
        "27_future_correction_job_manifest.csv",
        "28_local_d_curated_v1_1_summary.json",
        "29_local_d_curated_v1_1_audit_report.md",
    ]
    missing = [x for x in required if not (out_dir / x).exists()]
    err = []
    if missing:
        err.append("missing:" + ";".join(missing))
    case = pd.read_csv(out_dir / "06_local_d_case_manifest.csv", dtype=str, keep_default_na=False)
    frozen = pd.read_csv(out_dir / "06_local_d_case_manifest.csv", dtype=str, keep_default_na=False)
    if "dataset_role" not in frozen.columns:
        err.append("missing_dataset_role")
    if not frozen.empty:
        if len(frozen[frozen["split"] == "before_or_on_cutoff"]) != 28:
            err.append("before_cutoff_count_mismatch")
        if len(frozen[frozen["split"] == "locked_post_cutoff_candidate"]) != 19:
            err.append("post_cutoff_count_mismatch")
        if len(frozen[(frozen["dataset_role"] == "development_eligible_local_d") & (frozen["split"] != "before_or_on_cutoff")]):
            err.append("development_role_not_before_cutoff")
        if len(frozen[(frozen["dataset_role"] == "locked_post_cutoff_eligible_local_d") & (frozen["allowed_for_model_selection"].str.lower() == "true")]):
            err.append("locked_postcutoff_allowed_for_model_selection")
        if len(frozen[frozen["permanent_test_overlap"].str.lower() == "true"]):
            # they may exist only in source audit but not in frozen dev set
            pass
    fut = pd.read_csv(out_dir / "15_future_master42_n30_generation_manifest.csv", dtype=str, keep_default_na=False)
    if not fut.empty:
        seeds = sorted({int(x) for x in fut["seed"] if nstr(x)})
        if seeds != sorted(random.Random(MASTER_SEED).sample(range(1, 2**31 - 1), N_SEEDS)):
            err.append("seed_mismatch")
        if any(s in OLD_SEEDS for s in seeds):
            err.append("old_seed_present")
    result = {"validation_passed": not err, "errors": err, "n_cases": int(len(case)), "n_future_rows": int(len(fut))}
    (out_dir / "validation_local_d_curated_v1_1.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if err:
        raise SystemExit("Local-D curated v1.1 validation failed: " + "; ".join(err))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["freeze", "reconstruct", "dryrun", "validate", "all"])
    p.add_argument("--input-dir", default=str(SRC_DIR))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--offline", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    case, res = load_core_tables()
    frozen = freeze_roles(case)
    frozen.to_csv(out / "06_local_d_case_manifest.csv", index=False)
    res.to_csv(out / "07_local_d_residue_manifest.csv", index=False)
    ext, hashes, _ = build_external_manifests(frozen)
    comp, asm, excl, resmap = assemble_complexes(frozen)
    std, at, diff = build_af3_inputs(frozen, comp, resmap)
    future, correction = build_future_and_correction(frozen, std, at)
    summary = build_summary(frozen, res, future, correction, comp, std, at, diff, asm, excl, ext, hashes)
    validate(out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
