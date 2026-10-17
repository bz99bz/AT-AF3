# AT-AF3

**AT-AF3: Associative Thinking Enables Zero-Shot D-Protein Structure Generation**

This repository is the code and benchmark release for AT-AF3, a zero-shot
framework for controlling stereochemistry in AlphaFold 3 (AF3) predictions.
AF3 parameters are frozen. Associative Thinking (AT) compiles a chirality task
and residue-level chemical knowledge into AF3-compatible CCD constraints, and a
deterministic geometric verifier/corrector repairs admissible residual errors.
No task-specific training, fine-tuning or parameter updates are performed, and
experimental reference coordinates are withheld from generation and correction.

The benchmark is **CHORD** (*CHirality-Oriented Reference Dataset*): seven
All-D targets and nine Local-D targets, each evaluated with 30 matched AF3
inference seeds. The manuscript's reported results are 100% target D-chirality
accuracy on the nine Local-D targets and a mean All-D accuracy of 0.682 for the
complete AT-AF3 workflow; unmodified AF3 gives 0% on Local-D and 0.036 on All-D
under the same protocol.

## What is included

| Path | Contents |
|---|---|
| `scripts/` | All-D and Local-D pipelines, deterministic correction, GT metrics, WGE-3, table and figure scripts |
| `src/` | Reference-free chirality geometry, error-code construction, local mirror correction and structure loading |
| `configs/` | All-D target definitions, 30 matched master42 seeds, target-chirality configs |
| `config/` | Permanent-test denylist used by the benchmark-integrity guard |
| `data/gt_reference/` | Deposited reference mmCIFs used only for evaluation |
| `data/local_d/` | Local-D curation manifests and post-cutoff target manifests |
| `docs/` | Pipeline, metrics, data-provenance and figure-reproduction notes |
| `tests/` | Unit tests for input generation, chirality geometry and Phenix mapping |
| `visualization/` | Source data and plotting code for the manuscript's chirality, covalent, secondary and tertiary figures (run the scripts to generate rendered outputs) |

AF3 itself, its model parameters, Phenix, DockQ and USalign are **not**
redistributed. See [External software](#external-software).

## Installation

Python 3.10 or newer is required by the released scripts. The
`requirements.txt` file covers the released Python code; the AF3 and Phenix
environments have their own dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

### External software

| Tool | Used for | Configuration |
|---|---|---|
| AlphaFold 3 | frozen structure generation | `AF3_ROOT`, `AF3_PYTHON`, `AF3_MODEL_DIR` |
| Phenix/CCTBX | canonical PDB conversion, MolProbity, DSSP, bond restraints, WGE-3 | `PHENIX_PYTHON`, `PHENIX_MOLPROBITY`, `PHENIX_REDUCE` |
| DockQ | optional interface scoring | `DOCKQ` |
| USalign | optional official TM-score | `USALIGN` |

`AF3_ROOT` must contain `run_alphafold.py`; the default is
`<repository>/alphafold3`. `AF3_MODEL_DIR` is passed directly to AF3's
`--model_dir`. The AF3 checkout and model parameters must be obtained from
their own distribution under their own terms.

The release's `data/ccd_components_cache.json` contains the CCD chemistry
needed by WGE-3, so `components.cif.gz` is not required for the released
analyses.

## Quick start

### 1. All-D arm (7 targets × 30 seeds × 2 generation conditions)

```bash
export AF3_ROOT=/path/to/alphafold3
export AF3_PYTHON=/path/to/af3/venv/bin/python
export AF3_MODEL_DIR=/path/to/af3.bin
export PYTHON=python

bash scripts/run_all_d_pipeline.sh
```

This performs the following stages in order:

1. `build_master42_experiment_manifest.py` materializes 420 paired AF3/AT-AF3
   input JSONs and the generation manifest.
2. `run_master42_af3_at_af3_batch.py` runs frozen AF3 on Standard-AF3 and
   AT-AF3 inputs.
3. `build_master42_correction_manifest.py` binds each correction task to its
   AT-AF3 parent.
4. `run_master42_chirality_correction_batch.py` applies the deterministic
   correction to the 210 AT-AF3 structures.
5. `evaluate_master42_experiment.py` writes seed-level, case-level and paired
   bootstrap reports.
6. `ccdfix_table.py` writes the All-D target D-chirality table.

Use `FORCE=1` to rebuild the input/manifest tree, `SKIP_GENERATION=1` to reuse
existing AF3 outputs, and `MAX_PARALLEL=N` to control concurrency.

### 2. Local-D post-cutoff arm (9 targets × 30 seeds × 2 conditions + correction)

```bash
export AF3_PYTHON=/path/to/af3/venv/bin/python
export AF3_MODEL_DIR=/path/to/af3.bin
export AF3_ROOT=/path/to/alphafold3
bash scripts/run_local_d_pipeline.sh
```

The Local-D inputs are built only from `data/local_d/local_d_curated_v1_1`.
The nine post-cutoff targets are never used for fitting or model selection.

### 3. GT-referenced structural validation

After both arms have produced corrected structures:

```bash
export PHENIX_PYTHON=/path/to/phenix.python
export PHENIX_MOLPROBITY=/path/to/phenix.molprobity
export PHENIX_REDUCE=/path/to/phenix.reduce
export DOCKQ=/path/to/DockQ        # optional
export USALIGN=/path/to/USalign    # optional

bash scripts/run_structural_validation.sh
```

This writes canonical PDBs and MolProbity summaries under
`results/gt_comparison_phenix_ccdfix`, then runs the GT metrics, Cα RMSD,
DSSP/Q3, CCTBX bond-class and WGE-3 analyses. The suite reports target-site
D-chirality separately from distance-based metrics because lDDT and TM-score
are mirror-invariant.

### 4. Figures and tables

```bash
python scripts/plot_fig2_chirality_heatmap.py
```

Figure source packages and their measured input CSVs are under
`visualization/`. See `docs/FIGURE_REPRODUCTION.md`.


### 5. Tests

```bash
python -m pytest -q
```

The release test suite covers AF3 input construction, signed-volume
calibration, target-config loading for JSON and Local-D CSV manifests, structure
loading and the deterministic Phenix mapping helpers. It does not require AF3,
Phenix or a GPU. Phenix-dependent analysis scripts are exercised only under a
Phenix interpreter.

## Reproducibility contract

* **Seeds.** `configs/inference_seeds_master42.json` contains exactly the 30
  seeds produced by `random.Random(42).sample(range(1, 2**31 - 1), 30)`.
  `data/inference_seeds_master42.csv` is the same table in flat form.
* **Matched seeds.** All three conditions for a target use the same inference
  seed; no best-of-seed selection is performed.
* **No reference leakage.** `reference_coordinates_allowed` is `false` in every
  target config, and neither generation nor correction reads
  `data/gt_reference/`.
* **Deterministic correction.** The correction consumes the predicted
  structure and the target-chirality config only. It does not use experimental
  coordinates.
* **Frozen target configs.** The checked-in `configs/target_chirality/*.json`
  files are part of the benchmark definition; the manifest builder does not
  overwrite them. In particular, the all-D target chain is not always the
  alphabetically first chain in the deposited entry.
* **Frozen inputs.** The manifest and correction tasks record SHA-256 hashes of
  their input JSONs/structures.

See `docs/DATA_PROVENANCE.md` for target IDs, curation boundaries and generated
intermediate artefacts.

## Repository layout

```text
AT-AF3/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── requirements-dev.txt
├── configs/
│   ├── at_af_7.json
│   ├── inference_seeds_master42.json
│   └── target_chirality/*.json
├── data/
│   ├── gt_reference/{all_d,local_d}/*.cif
│   ├── local_d/
│   │   ├── pdb_local_d_*.csv
│   │   └── local_d_curated_v1_1/*.csv
│   ├── ccd_components_cache.json
│   └── inference_seeds_master42.csv
├── docs/
├── scripts/
│   ├── local_d_curated_v1/
│   ├── local_d_curated_v1_1/
│   ├── phenix_interaction_analysis/
│   └── ...
├── src/
├── tests/
└── visualization/
```

## Reproducing the manuscript's claims

The chirality endpoint is **target-site accuracy**, not whole-structure
success:

```text
signed volume = dot(cross(N − CA, C − CA), CB − CA)
D if negative, L if positive, ambiguous near zero, GLY excluded
```

A target site is evaluable only when N, CA, C and CB are present; missing and
degenerate sites are excluded from the denominator and recorded separately.
Target-level means use all 30 retained seeds, and benchmark summaries weight
targets equally. The reported statistical comparison is a two-sided Wilcoxon
signed-rank test on paired target means, not on residues or individual seeds.
Full definitions are in `docs/METRICS.md`.

Structural consistency is assessed at three levels: CCTBX/Engh–Huber bond
classes, DSSP Q3 composition, and WGE-3 interaction geometry. WGE-3 is
implemented in `scripts/phenix_interaction_analysis/`; it detects disulfide,
hydrogen-bond, hydrophobic and salt-bridge interactions in Phenix, measures
their characteristic distances and reports target-level deviations from the
experimental reference. It measures observed interaction geometry, not contact
recovery.

## Scope and limitations

* The release reproduces the frozen-AF3, zero-shot setting. It does not include
  an AF3 training pipeline.
* The correction is a deterministic geometric procedure, not a learned
  adapter.
* Distance-based structural metrics are chirality-blind; they must be read
  alongside the signed-volume endpoint.
* MolProbity categories such as Ramachandran, rotamer and Cβ deviations may be
  not evaluable for structures containing only non-standard/D residues; zero is
  not interpreted as perfect without evaluable-residue counts.
* The reference mmCIFs are included for evaluation only. They are not used by
  generation or correction.

## Citation

If you use this code or CHORD, please cite the AT-AF3 manuscript. Citation
metadata is in `CITATION.cff`.

## License

The release code is distributed under the MIT License; see `LICENSE`.
Third-party tools and reference structures retain their own terms.

