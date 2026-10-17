# Metrics

## 1. Target-site chirality

For every target residue with atoms N, CA, C and CB, the signed volume is

```
V = dot(cross(N - CA, C - CA), CB - CA)
```

with the documented calibration

* `V > 0` → L-like
* `V < 0` → D-like
* `|V| <= 1e-6` → ambiguous, not evaluable
* glycine → achiral and excluded

The calibration is checked with synthetic L-Ala and D-Ala by
`src.chirality_geometry.calibrate_chirality_sign()`; no experimental reference
is used. Atom order matters: swapping N and C changes the sign.

### Target-site accuracy

For target `p`, seed `k` and the evaluable target-site set `I_eval`:

```
A(p,k) = (1 / |I_eval|) * sum_i 1(current_i == target_i)
```

A site is evaluable only when N, CA, C and CB are present. Missing-atom and
degenerate sites are removed from the denominator for every condition and
recorded separately. `evaluate_master42_experiment.py` also returns the
unresolved count and the raw target count, so a change in the evaluable set
cannot improve a score without being recorded.

### Aggregation and tests

* The replication unit is the AF3 inference seed, never the residue.
* Target-level means use all 30 retained seeds.
* Benchmark summaries weight targets equally.
* Paired comparisons use the same seed across conditions.
* Confidence intervals in the seed-level evaluator use a paired seed bootstrap,
  not a residue bootstrap.
* The manuscript's condition comparison uses a two-sided Wilcoxon signed-rank
  test on paired target means (`n=7` All-D, `n=9` Local-D).

## 2. Structural metrics

### Covalent geometry

`analyze_bond_geometry.py` uses Phenix/CCTBX geometry-restraint bond proxies,
not distance thresholds. Automatic distance-based linking is disabled. For each
bond it records observed length, ideal length, restraint sigma, deviation and

```
Z = (d_observed - d_ideal) * sqrt(weight)
```

Summary metrics are bond RMSD (Å), bond RMSZ and the fraction of bonds with
`|Z| > 5`. The 11 Engh–Huber residue-context classes are defined in
`scripts/summarize_gt_bond_types_linux.py`.

### Secondary structure

`secondary_structure_q3.py` assigns DSSP states with Phenix `ksdssp` and
collapses them to

* H = {H, G, I}
* E = {E, B}
* C = everything else

Residues are matched within chains by sequence alignment, not raw residue
numbering. The output reports composition and Q3 agreement against GT; no
structure is mirrored.

### Tertiary structure

`tertiary_structure_rmsd.py` pairs chains through the canonical chain map and
residues by sequence alignment, then superposes each chain independently with
the Kabsch algorithm using proper rotations only. It reports the mean and
maximum chain Cα RMSD, matched-CA counts and radius of gyration. A separate
mirror-RMSD diagnostic is reported as chirality evidence, not as a fold score.

Some all-D entries deposit both the D and the L enantiomer of the target. For
those cases the config in `configs/target_chirality/` designates the chain that
holds the D copy; that chain is paired to the model by sequence, so the model is
scored against the D copy rather than against the L copy in the same entry.

`gt_metrics_suite.py` adds:

* per-chain TM-score, using the official USalign binary when available and a
  numpy implementation as a cross-check;
* all-atom lDDT at CASP thresholds 0.5/1/2/4 Å;
* site-neighbourhood lDDT for Local-D (default ±4 residues);
* matched-residue coverage;
* target-site D-chirality from canonical coordinates;
* optional DockQ interface scores.

TM-score and lDDT are distance-based and therefore mirror-invariant. They must
be read with the signed-volume endpoint, not as evidence of chirality.

### WGE-3 interaction geometry

`scripts/phenix_interaction_analysis/` detects disulfide, hydrogen-bond,
hydrophobic and salt-bridge interactions with Phenix. Hydrogens are added with
Phenix `reduce`; heavy-atom coordinates are left unchanged; no structure is
minimised, rebuilt or repaired. Every model is mapped to the reference by one
deterministic one-to-one chain assignment, with the declared target chain
taking priority.

For method `m`, target `p` and interaction class `t`:

```
delta(p,t,m) = 100 * |mean_distance(p,t,m) - mean_distance(p,t,GT)|
               / mean_distance(p,t,GT)
```

Unavailable measurements are left unreported. Because interaction inventories
can differ between a model and its reference, WGE-3 assesses observed
interaction geometry, not contact recovery.

## 3. Denominator and caveat policy

* Non-evaluable residues are excluded from the denominator and counted.
* Zero is not interpreted as a perfect score when a MolProbity category is not
  evaluable for D/non-standard residues.
* DockQ's default parser hides HETATM D residues; the release therefore uses
  chain/residue labels that preserve those residues for interface scoring and
  records residue coverage in `dockq_input_coverage.csv`.
* Mirror RMSD is reported as a signed-volume-adjacent diagnostic. A model can
  move away from the mirrored reference and still be far from the real target;
  both numbers must be shown together.
