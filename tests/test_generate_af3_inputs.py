import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_af3_inputs import (
    generate_baseline_input,
    generate_oracle_ccd_input,
    generate_prompt_conditioned_input,
    generate_all_inputs,
    parse_seed_args,
)


FIXTURE_CASES = [
    {
        "case_id": "case_03",
        "pdb_id": "7LL8",
        "case_type": "d_protein_binder",
        "sequence_binder": "EWKDHDGRSYVTSEFLRIDLGHHLRFRYLFPANPR",
        "target_chains": [
            {"chain_id": "B", "sequence": "SGQNHHEVVKFMDVYQRSYCHPIETLVDIFQEYPDEIEYIFKPSCVPLMRCGGCCNDEGLECVPTEESNITMQIMRIKPHQGQHIGEMSFLQHNKCECRPKKD"},
            {"chain_id": "C", "sequence": "SGQNHHEVVKFMDVYQRSYCHPIETLVDIFQEYPDEIEYIFKPSCVPLMRCGGCCNDEGLECVPTEESNITMQIMRIKPHQGQHIGEMSFLQHNKCECRPKKD"},
        ],
    },
    {
        "case_id": "case_04",
        "pdb_id": "8GQP",
        "case_type": "d_protein_binder",
        "sequence_binder": "LPVEKIIREAKKILDELLKRGLIDPELARIAREVLERARKLGNEEAARFVLELIERLRRELS",
        "target_chains": [
            {"chain_id": "B", "sequence": "DEHELLETAARWFYEIAKRA"},
        ],
    },
    {
        "case_id": "case_05",
        "pdb_id": "2R5D",
        "case_type": "d_peptide_binder",
        "sequence_binder": "HPCDYPEWQWLCELF",
        "target_chains": [
            {"chain_id": "B", "sequence": "RMKQIEDKIEEIESKQKKIENEIARIKKLLQLTVWGIKQLQARIL"},
            {"chain_id": "C", "sequence": "RMKQIEDKIEEIESKQKKIENEIARIKKLLQLTVWGIKQLQARIL"},
            {"chain_id": "D", "sequence": "RMKQIEDKIEEIESKQKKIENEIARIKKLLQLTVWGIKQLQARIL"},
        ],
    },
    {
        "case_id": "case_06",
        "pdb_id": "2Q33",
        "case_type": "all_D_mirror_image",
        "sequence": "ACG|DEFG",
        "chain_sequences": {"A": "ACG", "B": "DEFG"},
    },]


def case_by_id(case_id):
    return copy.deepcopy(next(case for case in FIXTURE_CASES if case["case_id"] == case_id))


def chain_ids(af3_input):
    return [entry["protein"]["id"] for entry in af3_input["sequences"]]


def modifications_by_chain(af3_input):
    return {
        entry["protein"]["id"]: entry["protein"].get("modifications", [])
        for entry in af3_input["sequences"]
    }


class GenerateAf3InputsTests(unittest.TestCase):
    def test_parse_explicit_seed_list_takes_precedence(self):
        args = type("Args", (), {
            "seeds": [101, 202, 303],
            "seed": 999,
            "seed_start": 1,
            "num_seeds": 2,
        })()
        self.assertEqual(parse_seed_args(args), [101, 202, 303])

    def test_parse_seed_range(self):
        args = type("Args", (), {
            "seeds": None,
            "seed": None,
            "seed_start": 101,
            "num_seeds": 3,
        })()
        self.assertEqual(parse_seed_args(args), [101, 102, 103])

    def test_parse_legacy_single_seed(self):
        args = type("Args", (), {
            "seeds": None,
            "seed": 101,
            "seed_start": None,
            "num_seeds": None,
        })()
        self.assertEqual(parse_seed_args(args), [101])

    def test_case_03_generates_a_b_c_chains(self):
        case = case_by_id("case_03")
        self.assertEqual(chain_ids(generate_baseline_input(case)), ["A", "B", "C"])
        self.assertEqual(chain_ids(generate_prompt_conditioned_input(case)), ["A", "B", "C"])

    def test_case_04_generates_a_b_chains(self):
        case = case_by_id("case_04")
        self.assertEqual(chain_ids(generate_baseline_input(case)), ["A", "B"])
        self.assertEqual(chain_ids(generate_prompt_conditioned_input(case)), ["A", "B"])

    def test_case_05_generates_a_b_c_d_chains(self):
        case = case_by_id("case_05")
        self.assertEqual(chain_ids(generate_baseline_input(case)), ["A", "B", "C", "D"])
        self.assertEqual(chain_ids(generate_prompt_conditioned_input(case)), ["A", "B", "C", "D"])


    def test_all_d_mirror_can_generate_two_chain_input(self):
        case = case_by_id("case_06")
        baseline = generate_baseline_input(case)
        prompt = generate_prompt_conditioned_input(case)
        self.assertEqual(chain_ids(baseline), ["A", "B"])
        self.assertEqual(chain_ids(prompt), ["A", "B"])
        mods = modifications_by_chain(prompt)
        self.assertEqual(len(mods["A"]), 2)  # ACG, glycine excluded
        self.assertEqual(len(mods["B"]), 3)  # DEFG, glycine excluded
    def test_only_binder_chain_has_d_modifications(self):
        case = case_by_id("case_03")
        mods = modifications_by_chain(generate_prompt_conditioned_input(case))
        self.assertTrue(mods["A"])
        self.assertEqual(mods["B"], [])
        self.assertEqual(mods["C"], [])

    def test_target_chains_have_no_d_modifications(self):
        case = case_by_id("case_05")
        mods = modifications_by_chain(generate_prompt_conditioned_input(case))
        self.assertTrue(mods["A"])
        for chain_id in ("B", "C", "D"):
            self.assertEqual(mods[chain_id], [])

    def test_missing_target_chains_raises_value_error(self):
        case = copy.deepcopy(case_by_id("case_03"))
        case.pop("target_chains")
        with self.assertRaises(ValueError):
            generate_baseline_input(case)

    def test_invalid_target_sequence_raises_value_error(self):
        case = copy.deepcopy(case_by_id("case_03"))
        case["target_chains"][0]["sequence"] = "VEGF-A (L-protein)"
        with self.assertRaises(ValueError):
            generate_baseline_input(case)

    def test_baseline_and_prompt_share_model_seed(self):
        case = case_by_id("case_04")
        baseline = generate_baseline_input(case)
        prompt = generate_prompt_conditioned_input(case)
        self.assertEqual(baseline["modelSeeds"], prompt["modelSeeds"])

    def test_oracle_ccd_all_d_uses_only_coarse_all_d_label(self):
        case = case_by_id("case_06")
        oracle = generate_oracle_ccd_input(case)
        prompt = generate_prompt_conditioned_input(case)
        self.assertEqual(oracle["name"], "2Q33_oracle_D")
        self.assertEqual(chain_ids(oracle), ["A", "B"])
        self.assertEqual(modifications_by_chain(oracle), modifications_by_chain(prompt))
        self.assertNotIn("at_prompt_sidecar", oracle)

    def test_oracle_ccd_binder_marks_only_the_binder_chain(self):
        case = case_by_id("case_04")
        oracle = generate_oracle_ccd_input(case)
        mods = modifications_by_chain(oracle)
        self.assertTrue(mods["A"])
        self.assertEqual(mods["B"], [])

    def test_oracle_ccd_local_d_does_not_use_target_positions(self):
        case = {
            "case_id": "case_local",
            "pdb_id": "LOCL",
            "case_type": "local_d_residue",
            "sequence": "ACWDE",
            "d_residue_positions": [3],
            "d_ccd_codes": ["DTR"],
        }
        oracle = generate_oracle_ccd_input(case)
        prompt = generate_prompt_conditioned_input(case)
        self.assertEqual(modifications_by_chain(oracle)["A"], [])
        self.assertEqual(modifications_by_chain(prompt)["A"], [{"ptmType": "DTR", "ptmPosition": 3}])


    def test_generate_all_inputs_cleans_stale_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            stale_baseline = out / "baseline" / "stale.json"
            stale_prompt = out / "prompt_conditioned" / "stale.json"
            stale_baseline.parent.mkdir(parents=True, exist_ok=True)
            stale_prompt.parent.mkdir(parents=True, exist_ok=True)
            stale_baseline.write_text("stale")
            stale_prompt.write_text("stale")
            (out / "af3_inputs_manifest.json").write_text("[]")

            generate_all_inputs([case_by_id("case_03")], output_base=out)

            self.assertFalse(stale_baseline.exists())
            self.assertFalse(stale_prompt.exists())
            self.assertTrue((out / "baseline" / "7LL8_baseline.json").exists())
            self.assertTrue((out / "prompt_conditioned" / "7LL8_prompt_D.json").exists())
            manifest = json.loads((out / "af3_inputs_manifest.json").read_text())
            self.assertEqual(len(manifest), 1)


if __name__ == "__main__":
    unittest.main()
