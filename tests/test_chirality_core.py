"""Unit tests for the reference-free chirality geometry and target configs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.chirality_error_code import build_error_code_rows, load_target_config
from src.chirality_geometry import (
    calibrate_chirality_sign,
    classify_residue_chirality,
    ensure_chirality_sign_mapping,
    residue_atom_map,
    signed_chiral_volume,
)
from src.structure_atom_loader import load_structure_atoms


def ala_atoms(cb_z: float) -> list[dict]:
    """One alanine with a chosen CB z coordinate (L if cb_z > 0)."""
    coords = {
        "N": (1.0, 0.0, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (0.0, 1.0, 0.0),
        "O": (0.0, 1.5, 0.0),
        "CB": (0.0, 0.0, cb_z),
    }
    return [
        {
            "serial": index,
            "name": name,
            "resname": "ALA",
            "chain": "A",
            "resid": 1,
            "x": x,
            "y": y,
            "z": z,
            "element": name[0],
        }
        for index, (name, (x, y, z)) in enumerate(coords.items(), start=1)
    ]


class ChiralityGeometryTests(unittest.TestCase):
    def test_signed_volume_uses_documented_atom_order(self):
        l_volume = signed_chiral_volume(residue_atom_map(ala_atoms(1.0), "A", 1))
        d_volume = signed_chiral_volume(residue_atom_map(ala_atoms(-1.0), "A", 1))
        self.assertGreater(l_volume, 0.0)
        self.assertLess(d_volume, 0.0)
        self.assertAlmostEqual(l_volume, -d_volume)

    def test_classification_is_positive_l_negative_d(self):
        self.assertEqual(classify_residue_chirality(ala_atoms(1.0))["chirality"], "L")
        self.assertEqual(classify_residue_chirality(ala_atoms(-1.0))["chirality"], "D")

    def test_near_planar_center_is_not_evaluable(self):
        result = classify_residue_chirality(ala_atoms(0.0))
        self.assertEqual(result["chirality_status"], "ambiguous")

    def test_glycine_is_achiral(self):
        atoms = [dict(atom, resname="GLY") for atom in ala_atoms(1.0)]
        result = classify_residue_chirality(atoms)
        self.assertEqual(result["chirality_status"], "achiral")
        self.assertEqual(result["reason"], "glycine_no_cb")

    def test_calibration_and_mapping_file(self):
        calibration = calibrate_chirality_sign()
        self.assertEqual(calibration["positive_sign"], "L")
        self.assertEqual(calibration["negative_sign"], "D")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "sign.json"
            ensure_chirality_sign_mapping(path)
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["signed_volume_formula"], calibration["signed_volume_formula"])


class TargetConfigTests(unittest.TestCase):
    def test_all_d_json_config(self):
        config = load_target_config(ROOT / "configs/target_chirality/7P5R_2.json", case_id="7P5R_2")
        self.assertEqual(config["chains"]["C"]["default_target_chirality"], "D")
        self.assertFalse(config["reference_coordinates_allowed"])

    def test_local_d_csv_config_for_curated_target(self):
        config = load_target_config(
            ROOT / "data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv",
            case_id="1GO9_1",
        )
        self.assertIn("A", config["chains"])
        self.assertTrue(config["chains"]["A"]["residue_targets"])

    def test_error_rows_are_reference_free(self):
        config = load_target_config(ROOT / "configs/target_chirality/7QDI_1.json", case_id="7QDI_1")
        rows = build_error_code_rows(ala_atoms(1.0), config, "7QDI_1", seed=1)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("reference", rows[0])


class PromptTargetTests(unittest.TestCase):
    def test_actual_all_d_and_heterochiral_cases_are_resolved(self):
        from associative_chirality_memory import resolve_chirality_targets

        cases = json.loads((ROOT / "configs/at_af_7.json").read_text())["cases"]
        by_id = {case["pdb_entity_id"]: case for case in cases}
        all_d = resolve_chirality_targets(by_id["7P5R_2"])
        self.assertEqual(all_d["mode"], "all_d_mirror")
        self.assertTrue(all_d["d_positions"])
        binder = resolve_chirality_targets(by_id["8GQP_1"])
        self.assertEqual(binder["mode"], "heterochiral_complex")
        self.assertEqual(binder["d_chains"], ["A"])
        self.assertEqual(binder["l_chains"], ["B", "D"])
        self.assertEqual(binder["expected_labels"]["B"], "L")
        self.assertTrue(binder["d_positions"])


class StructureLoaderTests(unittest.TestCase):
    def test_pdb_loader_reads_chain_residue_and_coordinates(self):
        pdb = (
            "ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       0.000   1.000   0.000  1.00  0.00           C\n"
            "ATOM      4  CB  ALA A   1       0.000   0.000   1.000  1.00  0.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ala.pdb"
            path.write_text(pdb, encoding="utf-8")
            atoms, meta = load_structure_atoms(path)
        self.assertTrue(meta["parse_success"])
        self.assertEqual(len(atoms), 4)
        self.assertEqual(atoms[0]["chain"], "A")
        self.assertEqual(atoms[0]["resid"], 1)
        self.assertEqual(classify_residue_chirality(atoms)["chirality"], "L")


if __name__ == "__main__":
    unittest.main()
