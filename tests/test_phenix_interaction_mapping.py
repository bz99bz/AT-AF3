"""Invariants of the GT <-> model chain correspondence (``phenix_interaction_analysis.mapping``).

The correspondence is the backbone of the whole comparison - every interaction is carried from
the reference onto a model through it - so its *rules* are pinned here on small, hand-built
structures, independently of any interaction detection:

1. the assignment is the global optimum of the parent-chemistry agreement, not a per-chain
   argmax: no GT chain is ever used by two model chains
   (``test_global_assignment_beats_the_per_chain_argmax``);
2. it is deterministic and independent of the order the chains appear in the files, and it never
   depends on interaction scores or on geometry - only on the residue chemistry
   (``test_deterministic_and_order_independent``, ``test_geometry_never_influences_the_pairing``);
3. the declared **target** chain - the reference copy the case configuration names - claims the
   model chain it reproduces best, so a second copy of the same molecule that happens to be better
   resolved cannot take the model away from it (``test_declared_target_claims_its_model_chain``,
   ``test_declared_target_takes_the_model_chain_it_reproduces_best``); only the declared reference
   scope and the chain ids break ties between equally agreeing pairs, and no declaration can create
   a pair the chemistry contradicts (``test_declared_target_breaks_a_chemistry_tie``,
   ``test_declaration_cannot_create_a_pair_the_chemistry_rejects``);
4. the D/L character is an *outcome*, never an input, so the uncorrected and the corrected model of
   a case receive the same correspondence (``test_chirality_is_an_outcome_not_an_input``);
5. unresolved parent chemistry is one unspecified class, so the same unresolved modification on
   both sides still matches and a chirality-corrected copy keeps the same correspondence
   (``test_unresolved_parent_chemistry_is_one_class``);
6. a chain without a compatible partner is reported unmatched instead of being forced onto an
   unrelated reference chain, and the reason is visible in ``chain_mapping.csv``
   (``test_unmatched_chains_are_reported``, ``CompatibilityGateTest``);
7. the residue-level correspondence follows the chain assignment, keeps insertion codes and
   reports every reference residue that has no counterpart (``test_residue_level_correspondence``);
8. the mapping output names both sides explicitly (``reference_chain`` / ``model_chain``) and
   classifies every unselected reference chain by its declared role
   (``ReferenceRoleClassificationTest``): a copy outside the declared scope - 8BFD_2 chain A - is
   ``required=no`` and is never counted as a missing required chain.

Run with the interpreter of the analysis (no Phenix needed for these tests):
``python3 -m unittest tests.test_phenix_interaction_mapping``.
"""
from __future__ import annotations

import itertools
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phenix_interaction_analysis import chem as C          # noqa: E402
from phenix_interaction_analysis import mapping as M       # noqa: E402
from phenix_interaction_analysis import protocol as P      # noqa: E402
from phenix_interaction_analysis.structure_io import (Atom, Chain,  # noqa: E402
                                                       Residue, Structure)

RESOLVER = C.ChemResolver()          # standard amino acids + curated D table; no CCD cache needed


def residue(chain_id: str, seq: int, name: str, xyz=(0.0, 0.0, 0.0), icode: str = "") -> Residue:
    """One residue with the four backbone heavy atoms; chemistry resolved like the real pipeline."""
    atoms = [Atom(atom_name, element, xyz, 1.0, "", False)
             for atom_name, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))]
    return Residue(chain=chain_id, seq=str(seq), icode=icode, resname=name,
                   chem=RESOLVER.resolve(name), is_het=False, atoms=atoms)


def unsupported(chain_id: str, seq: int, name: str, is_d: bool = False) -> Residue:
    """A peptide residue whose CCD parent is not a standard amino acid (chem_class == "")."""
    info = C.ChemInfo(resname=name, chem_class="", parent="", is_D=is_d, is_polymer=True,
                      is_supported=False, source="ccd")
    return Residue(chain=chain_id, seq=str(seq), icode="", resname=name, chem=info, is_het=False,
                   atoms=[Atom("N", "N", (0.0, 0.0, 0.0), 1.0, "", False)])


def chain(chain_id: str, names) -> Chain:
    return Chain(id=chain_id,
                 residues=[residue(chain_id, i + 1, name) for i, name in enumerate(names)])


def structure(name: str, *chains) -> Structure:
    return Structure(name=name, source_path=f"{name}.pdb", chains=list(chains))


def mapping_of(model, gt, target_chain: str = "", declared=(), enforce=True) -> dict:
    """``{model chain: gt chain or None}`` as the pipeline consumes it."""
    return {cm.model_chain: (cm.reference_chain or None)
            for cm in M.map_chains(model, gt, target_chain, declared, enforce)}


class GlobalAssignmentTest(unittest.TestCase):
    """One-to-one, globally optimal, never a per-chain argmax (no chain reuse)."""

    def setUp(self):
        self.gt = structure("gt", chain("X", ["ALA"] * 10), chain("Y", ["VAL"] * 10))
        # A agrees with X on 10 positions and with Y on 10 positions; B agrees with X on 9.
        self.model = structure("model", chain("A", ["ALA"] * 10 + ["VAL"] * 10),
                               chain("B", ["ALA"] * 9 + ["GLY"] * 9))

    def test_global_assignment_beats_the_per_chain_argmax(self):
        # the per-chain argmax (the old behaviour) sends *both* chains to X: 10 + 9 with X reused
        cmaps = M.map_chains(self.model, self.gt)
        chosen = {cm.model_chain: cm.reference_chain for cm in cmaps if cm.matched}
        self.assertEqual(chosen, {"A": "Y", "B": "X"})
        self.assertEqual(sorted(chosen), ["A", "B"])

    def test_no_gt_chain_is_used_twice(self):
        partners = [cm.reference_chain for cm in M.map_chains(self.model, self.gt) if cm.matched]
        self.assertEqual(len(partners), len(set(partners)))

    def test_every_model_chain_appears_exactly_once(self):
        cmaps = M.map_chains(self.model, self.gt)
        ids = [cm.model_chain for cm in cmaps]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(set(ids), {"A", "B"})

    def test_hungarian_solver_returns_a_maximum_weight_assignment(self):
        """The solver behind the assignment must be optimal on random matrices (brute force)."""
        rng = random.Random(20260101)
        for size in range(1, 7):
            for _ in range(40):
                weights = [[rng.randrange(0, 25) for _ in range(size)] for _ in range(size)]
                columns = M._maximum_assignment([row[:] for row in weights])
                self.assertEqual(sorted(columns), list(range(size)))
                got = sum(weights[row][columns[row]] for row in range(size))
                best = max(sum(weights[row][perm[row]] for row in range(size))
                           for perm in itertools.permutations(range(size)))
                self.assertEqual(got, best)

    def test_hungarian_solver_handles_rectangular_input(self):
        """Padding to a square matrix must not lose the assignment of the real rows."""
        wide = [[5, 5, 5], [5, 0, 0]]              # more columns than rows
        columns = M._maximum_assignment(wide)
        self.assertEqual(len(columns), 3)          # one column per row of the padded square
        real = columns[:len(wide)]
        self.assertEqual(len(set(real)), len(wide))
        best = max(sum(wide[row][perm[row]] for row in range(len(wide)))
                   for perm in itertools.permutations(range(3), len(wide)))
        self.assertEqual(best, 10)
        self.assertEqual(sum(wide[row][real[row]] for row in range(len(wide))), best)
        tall = [[5, 1], [2, 2], [1, 5]]            # more rows than columns
        columns = M._maximum_assignment(tall)
        self.assertEqual(len(columns), 3)
        self.assertEqual(sum(tall[row][column] for row, column in enumerate(columns)
                             if column < 2), 10)

    def test_deterministic_and_order_independent(self):
        first = mapping_of(self.model, self.gt)
        self.assertEqual(first, mapping_of(self.model, self.gt))
        shuffled_model = structure("model", *reversed(self.model.chains))
        shuffled_gt = structure("gt", *reversed(self.gt.chains))
        self.assertEqual(first, mapping_of(shuffled_model, shuffled_gt))

    def test_geometry_never_influences_the_pairing(self):
        """Different coordinates, altlocs and even a different atom inventory: same mapping."""
        far = structure("model",
                        Chain(id="A", residues=[residue("A", i + 1, n, xyz=(90.0 * i, 0.0, 0.0))
                                                for i, n in enumerate(["ALA"] * 10 + ["VAL"] * 10)]),
                        Chain(id="B", residues=[residue("B", i + 1, n, xyz=(0.0, 90.0 * i, 0.0))
                                                for i, n in enumerate(["ALA"] * 9 + ["GLY"] * 9)]))
        for res in far.chains[0].residues:
            res.atoms = res.atoms[:1]                      # side chain not resolved
        self.assertEqual(mapping_of(self.model, self.gt), mapping_of(far, self.gt))

    def test_mapping_module_takes_no_score_or_distance_input(self):
        import inspect
        signature = inspect.signature(M.map_chains)
        self.assertEqual(list(signature.parameters), ["model", "gt", "target_chain",
                                                     "declared_chains",
                                                     "enforce_declared_target"])

    def test_alternative_mode_reproduces_the_copy_that_won_on_raw_length(self):
        """``enforce_declared_target=False`` selects the alternative convention.

        Same miniature as above: with the switch off the declared copy Y no longer claims the model
        chain and the better resolved copy X takes it.  The default keeps the declared-target rule.
        """
        gt = structure("gt", chain("X", ["ALA"] * 12 + ["VAL"] * 11), chain("Y", ["ALA"] * 12))
        model = structure("model", chain("A", ["ALA"] * 12 + ["VAL"] * 11))
        self.assertEqual(mapping_of(model, gt, "Y", ("Y",)), {"A": "Y"})
        self.assertEqual(mapping_of(model, gt, "Y", ("Y",), enforce=False), {"A": "X"})
        self.assertEqual(mapping_of(model, gt, "Y", ("Y",), enforce=True), {"A": "Y"})

    def test_alternative_mode_still_keeps_the_declaration_as_a_tie_break(self):
        """The alternative convention keeps the declared copy as a tie-break between equal copies."""
        gt = structure("gt", chain("X", ["ALA"] * 10), chain("Y", ["ALA"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10))
        self.assertEqual(mapping_of(model, gt, "Y", ("Y",), enforce=False), {"A": "Y"})


class DeclarationTest(unittest.TestCase):
    """The declared target chain claims its copy; the scope and the chain ids only break ties."""

    def test_declared_target_breaks_a_chemistry_tie(self):
        gt = structure("gt", chain("X", ["ALA"] * 12), chain("Y", ["ALA"] * 12))
        model = structure("model", chain("A", ["ALA"] * 12 + ["VAL"] * 11))
        self.assertEqual(mapping_of(model, gt)["A"], "X")            # chain id: none matches -> X
        self.assertEqual(mapping_of(model, gt, "Y")["A"], "Y")       # declared target wins the tie
        self.assertEqual(mapping_of(model, gt, "", ("Y",))["A"], "Y")  # declared scope wins it

    def test_declaration_cannot_create_a_pair_the_chemistry_rejects(self):
        # the declared target X agrees on 11 positions but covers only 48 % of the model chain, so
        # it is not compatible; the declaration must not rescue it
        gt = structure("gt", chain("X", ["VAL"] * 11), chain("Y", ["ALA"] * 12))
        model = structure("model", chain("A", ["ALA"] * 12 + ["VAL"] * 11))
        self.assertEqual(mapping_of(model, gt, "X", ("X", "Y"))["A"], "Y")

    def test_chain_id_is_the_fallback_tie_break(self):
        gt = structure("gt", chain("B", ["ALA"] * 10), chain("A", ["ALA"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10))
        self.assertEqual(mapping_of(model, gt)["A"], "A")

    def test_declared_target_claims_its_model_chain(self):
        """The declared copy takes the model chain even when another copy agrees one position more.

        8BFD_2 in miniature: the entry holds two copies of the same molecule and the configured
        target chain - the copy carrying the real D target - has one residue less resolved, so the
        other copy agrees with the model on one identical position more.  A reference copy's
        resolved length is a completeness artefact, not a statement about the model, and the
        configuration names the copy the model has to be scored against: the *declared* copy takes
        the model chain and the better resolved copy stays unmatched.  Without the declaration -
        and with a declared scope, which is a tie-break only - the chemistry still decides.
        """
        gt = structure("gt", chain("X", ["ALA"] * 12 + ["VAL"] * 11), chain("Y", ["ALA"] * 12))
        model = structure("model", chain("A", ["ALA"] * 12 + ["VAL"] * 11))
        self.assertEqual(mapping_of(model, gt, "Y", ("Y",))["A"], "Y")
        self.assertEqual(mapping_of(model, gt, "", ("Y",))["A"], "X")
        self.assertEqual(mapping_of(model, gt)["A"], "X")

    def test_declared_target_takes_the_model_chain_it_reproduces_best(self):
        """The target claims the single model chain it reproduces best - and no other one."""
        gt = structure("gt", chain("X", ["ALA"] * 10), chain("Y", ["VAL"] * 10))
        model = structure("model", chain("A", ["VAL"] * 10), chain("B", ["ALA"] * 10))
        self.assertEqual(mapping_of(model, gt, "X", ("X",)), {"A": "Y", "B": "X"})

    def test_declared_target_prefers_the_model_chain_with_its_own_id(self):
        """Between equally agreeing model chains the claim takes the one with the target's id."""
        gt = structure("gt", chain("Z", ["ALA"] * 10), chain("A", ["ALA"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10), chain("Z", ["ALA"] * 10))
        cmaps = {cm.model_chain: cm for cm in M.map_chains(model, gt, "Z")}
        self.assertEqual(cmaps["Z"].basis, "sequence+target_chain+chain_id")   # claimed, not "A"
        self.assertEqual(cmaps["A"].reference_chain, "A")


class ChiralityTest(unittest.TestCase):
    """D/L is an outcome of the comparison, never an input to it."""

    def test_chirality_is_an_outcome_not_an_input(self):
        gt = structure("gt", chain("A", ["DAL"] * 10))               # all-D reference
        model_d = structure("model_d", chain("A", ["DAL"] * 10))     # uncorrected, all-D
        model_l = structure("model_l", chain("A", ["ALA"] * 10))     # chirality-corrected, all-L
        self.assertEqual(mapping_of(model_d, gt), {"A": "A"})
        self.assertEqual(mapping_of(model_l, gt), mapping_of(model_d, gt))
        delta = {cm.model_chain: cm.d_fraction_delta for cm in M.map_chains(model_d, gt)}
        delta_l = {cm.model_chain: cm.d_fraction_delta for cm in M.map_chains(model_l, gt)}
        self.assertEqual(delta["A"], 0.0)
        self.assertEqual(delta_l["A"], 1.0)                # reported, and it changed nothing else

    def test_dl_character_does_not_choose_between_two_copies(self):
        """Two GT copies differing only in chirality: the declared target picks, not the D count."""
        gt = structure("gt", chain("A", ["ALA"] * 10), chain("B", ["DAL"] * 10))
        model = structure("model", chain("A", ["DAL"] * 10))
        self.assertEqual(mapping_of(model, gt)["A"], "A")            # chain id outranks the D count
        self.assertEqual(mapping_of(model, gt, "B")["A"], "B")

    def test_unresolved_parent_chemistry_is_one_class(self):
        """An unresolved modification matches the *same* unresolved position on the other side.

        The CCD parent of these residues is not a standard amino acid, so no D/L-invariant parent
        can be derived; treating them as one unspecified class keeps the correspondence of a chain
        whose only non-standard residues are unresolved (the corpus has at most three per chain)
        and keeps the uncorrected and the corrected model paired.
        """
        gt = structure("gt", Chain(id="A", residues=[unsupported("A", i + 1, "28J", is_d=True)
                                                     for i in range(10)]))
        model = structure("model", Chain(id="A", residues=[unsupported("A", i + 1, "UNK")
                                                           for i in range(10)]))
        self.assertEqual(mapping_of(model, gt), {"A": "A"})


class CompatibilityGateTest(unittest.TestCase):
    """Incompatible pairs become unmatched - never a forced correspondence."""

    def test_too_few_identical_residues_is_unmatched(self):
        gt = structure("gt", chain("A", ["ALA"] * 2))
        model = structure("model", chain("A", ["ALA"] * 2))
        self.assertEqual(mapping_of(model, gt), {"A": None})
        self.assertEqual(M.map_chains(model, gt)[0].basis, "no_compatible_reference_chain")

    def test_high_identity_on_a_minority_of_the_model_chain_is_unmatched(self):
        gt = structure("gt", chain("A", ["ALA"] * 4))
        model = structure("model", chain("A", ["ALA"] * 4 + ["GLY"] * 6))
        self.assertEqual(mapping_of(model, gt), {"A": None})

    def test_gate_boundary_is_inclusive(self):
        gt = structure("gt", chain("A", ["ALA"] * 4))
        model = structure("model", chain("A", ["ALA"] * 4 + ["GLY"] * 4))
        self.assertEqual(mapping_of(model, gt), {"A": "A"})

    def test_a_few_extra_residues_do_not_break_the_pair(self):
        """A model chain with a few extra residues (a tag, unresolved ends) is still its reference.

        The canonical models carry unresolved residues at both ends (the 7QDI_1 model is literally
        ``UNK + reference + UNK``); that costs coverage but must not cost the pair.
        """
        gt = structure("gt", chain("A", ["ALA"] * 20))
        model = structure("model", chain("A", ["GLY"] * 4 + ["ALA"] * 20))
        self.assertEqual(mapping_of(model, gt), {"A": "A"})
        tagged = structure("model", Chain(id="A", residues=(
            [unsupported("A", 1, "UNK")]
            + [residue("A", i + 2, "ALA") for i in range(20)]
            + [unsupported("A", 22, "UNK")])))
        self.assertEqual(mapping_of(tagged, gt), {"A": "A"})

    def test_a_short_reference_chain_cannot_claim_a_long_model_chain(self):
        """The gate is a safety net against a coincidental window (8GQP_1 A vs B: 12/62/19).

        Twelve identical positions clear ``MIN_PAIR_RESIDUES`` but explain only 12 of the model
        chain's 62 residues, so the reference chain is reported as unmatched rather than forcing
        the model chain onto it.
        """
        gt = structure("gt", chain("A", ["GLY"] * 12 + ["TRP"] * 7))
        model = structure("model", chain("A", ["GLY"] * 12 + ["SER"] * 50))
        self.assertEqual(mapping_of(model, gt), {"A": None})
        self.assertEqual(M.map_chains(model, gt)[0].basis, "no_compatible_reference_chain")

    def test_unmatched_chains_are_reported(self):
        gt = structure("gt", chain("X", ["ALA"] * 10), chain("Y", ["VAL"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10), chain("Z", ["GLY"] * 10))
        cmaps = M.map_chains(model, gt)
        self.assertEqual({cm.model_chain: cm.reference_chain for cm in cmaps if cm.matched},
                         {"A": "X"})
        rows = M.chain_mapping_rows(gt, cmaps)
        by_key = {(r["model_chain"], r["reference_chain"]): r for r in rows}
        unmatched_model = by_key[("Z", P.NA)]
        self.assertEqual(unmatched_model["matched"], "no")
        self.assertEqual(unmatched_model["assignment_basis"], "no_compatible_reference_chain")
        self.assertEqual(unmatched_model["identical_residues"], 0)
        unmatched_gt = by_key[(P.NA, "Y")]
        self.assertEqual(unmatched_gt["assignment_basis"], "no_compatible_model_chain")
        self.assertEqual(unmatched_gt["identical_residues"], P.NA)
        self.assertEqual(len(rows), 3)                    # A->X, Z unmatched, Y unmatched
        summary = M.correspondence_summary(gt, cmaps)
        self.assertIn("reference X <- model A", summary)
        self.assertIn("Z", summary)
        self.assertIn("unselected reference chain(s) Y", summary)


class ReferenceRoleClassificationTest(unittest.TestCase):
    """``chain_mapping.csv`` names both sides and classifies every unselected reference chain.

    8BFD_2 in miniature: the entry deposits two copies of the same molecule, the case configuration
    declares only the D copy (chain X) and the model reproduces the declared copy.  The other copy
    (chain Y, the L enantiomer) is *not* required, so its ``chain_mapping.csv`` row is
    ``outside_declared_scope`` / ``required=no`` - it must never be counted as a missing required
    chain.  A declared chain that really was left unselected stays ``required=yes``.
    """

    def test_reference_chain_columns_are_explicit(self):
        gt = structure("gt", chain("B", ["ALA"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10))
        row = M.chain_mapping_rows(gt, M.map_chains(model, gt, "B", ("B",)), "B", ("B",))[0]
        self.assertEqual(row["reference_chain"], "B")
        self.assertEqual(row["model_chain"], "A")
        self.assertNotIn("gt_chain", row)

    def test_unselected_copy_outside_the_declared_scope_is_not_required(self):
        gt = structure("gt", chain("X", ["ALA"] * 12 + ["VAL"] * 11), chain("Y", ["ALA"] * 12))
        model = structure("model", chain("A", ["ALA"] * 12 + ["VAL"] * 11))
        cmaps = M.map_chains(model, gt, "X", ("X",))
        rows = M.chain_mapping_rows(gt, cmaps, "X", ("X",))
        by_chain = {(r["reference_chain"], r["model_chain"]): r for r in rows}
        selected = by_chain[("X", "A")]
        self.assertEqual(selected["reference_role"], M.ROLE_DECLARED_TARGET)
        self.assertEqual(selected["required"], "yes")
        self.assertEqual(selected["selection"], M.SELECTED)
        unselected = by_chain[("Y", P.NA)]
        self.assertEqual(unselected["matched"], "no")
        self.assertEqual(unselected["reference_role"], M.ROLE_OUTSIDE_DECLARED_SCOPE)
        self.assertEqual(unselected["required"], "no")
        self.assertEqual(unselected["selection"], M.UNSELECTED_OUTSIDE_SCOPE)
        self.assertEqual(unselected["assignment_basis"], "outside_declared_scope")
        summary = M.correspondence_summary(gt, cmaps, "X", ("X",))
        self.assertIn("unselected reference chain(s) Y", summary)
        self.assertIn("not required, not a missing chain", summary)

    def test_declared_chain_left_unselected_stays_required(self):
        """A declared scope chain with no compatible model chain is a real gap, not a scope note."""
        gt = structure("gt", chain("X", ["GLY"] * 12 + ["TRP"] * 7), chain("Y", ["ALA"] * 21))
        model = structure("model", chain("A", ["ALA"] * 21))
        cmaps = M.map_chains(model, gt, "Y", ("X", "Y"))
        self.assertEqual(M.map_chains(model, gt, "Y", ("X", "Y"))[0].reference_chain, "Y")
        rows = M.chain_mapping_rows(gt, cmaps, "Y", ("X", "Y"))
        unselected = [r for r in rows if r["reference_chain"] == "X"]
        self.assertEqual(len(unselected), 1)
        self.assertEqual(unselected[0]["reference_role"], M.ROLE_DECLARED_SCOPE)
        self.assertEqual(unselected[0]["required"], "yes")
        self.assertEqual(unselected[0]["selection"], M.UNSELECTED_REQUIRED)
        self.assertEqual(unselected[0]["assignment_basis"], "no_compatible_model_chain")

    def test_case_without_a_declaration_has_no_chain_outside_its_scope(self):
        """The Local-D cases declare nothing: every reference chain is in scope (required)."""
        gt = structure("gt", chain("A", ["ALA"] * 10), chain("B", ["VAL"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10))
        rows = M.chain_mapping_rows(gt, M.map_chains(model, gt), "", ())
        unselected = [r for r in rows if r["reference_chain"] == "B"]
        self.assertEqual(len(unselected), 1)
        self.assertEqual(unselected[0]["reference_role"], M.ROLE_IN_SCOPE_UNDECLARED)
        self.assertEqual(unselected[0]["required"], "yes")


class ResidueLevelCorrespondenceTest(unittest.TestCase):
    """The residue pairs follow the chain assignment and keep the reference residues visible."""

    def _coded_chain(self, chain_id, entries):
        return Chain(id=chain_id,
                     residues=[residue(chain_id, seq, name, icode=icode)
                               for seq, icode, name in entries])

    def test_forward_and_reverse_maps_follow_the_chain_assignment(self):
        # distinct parent chemistries keep the alignment unambiguous, so the 4-residue model chain
        # is the reference's first four residues and the last two stay reference-only
        gt = structure("gt", chain("A", ["ALA", "GLY", "SER", "THR", "VAL", "LEU"]))
        model = structure("model", chain("A", ["ALA", "GLY", "SER", "THR"]))
        cmaps = M.map_chains(model, gt, target_chain="A")
        rows, forward, reverse = M.residue_mapping(model, gt, cmaps, "A")
        self.assertEqual(forward[("A", "A:1")], ("A", "A:1"))
        self.assertEqual(reverse[("A", "A:4")], ("A", "A:4"))
        mapped = [r for r in rows if r["mapped"] == "yes"]
        self.assertEqual(len(mapped), 4)
        self.assertTrue(all(r["is_target_chain"] == "yes" for r in rows))
        gt_only = [r["gt_residue"] for r in rows if r["mapped"] == "no"]
        self.assertEqual(sorted(gt_only), ["A:5", "A:6"])

    def test_indistinguishable_residues_are_paired_deterministically(self):
        """A run of identical parent chemistry ties everywhere; the pairs stay a deterministic
        bijection onto a contiguous block of the reference."""
        gt = structure("gt", chain("A", ["ALA"] * 6))
        model = structure("model", chain("A", ["ALA"] * 4))
        cmaps = M.map_chains(model, gt)
        first = M.residue_mapping(model, gt, cmaps, "")
        second = M.residue_mapping(model, gt, cmaps, "")
        self.assertEqual(first, second)
        _rows, forward, _reverse = first
        self.assertEqual(len(forward), 4)
        self.assertEqual(len({tok for _chain, tok in forward.values()}), 4)   # bijective
        reference_tokens = sorted(int(tok.split(":")[1]) for _chain, tok in forward.values())
        self.assertEqual(reference_tokens, list(range(reference_tokens[0],
                                                       reference_tokens[0] + 4)))

    def test_insertion_codes_are_preserved(self):
        gt = structure("gt", self._coded_chain("A", [(52, "", "ALA"), (52, "A", "ALA"),
                                                     (52, "B", "ALA"), (53, "", "ALA")]))
        model = structure("model", self._coded_chain("A", [(52, "", "ALA"), (52, "A", "ALA"),
                                                           (52, "B", "ALA"), (53, "", "ALA")]))
        rows, forward, _ = M.residue_mapping(model, gt, M.map_chains(model, gt), "")
        self.assertEqual(len([r for r in rows if r["mapped"] == "yes"]), 4)
        self.assertEqual(forward[("A", "A:52B")], ("A", "A:52B"))

    def test_unmatched_chains_produce_no_residue_rows(self):
        gt = structure("gt", chain("A", ["ALA"] * 10))
        model = structure("model", chain("A", ["ALA"] * 10), chain("Z", ["GLY"] * 5))
        cmaps = M.map_chains(model, gt)
        rows, forward, reverse = M.residue_mapping(model, gt, cmaps, "")
        self.assertFalse([r for r in rows if r["model_chain"] == "Z"])
        self.assertFalse([k for k in reverse if k[0] == "Z"])
        self.assertTrue(forward)


