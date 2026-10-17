# Data provenance

## CHORD

CHORD (*CHirality-Oriented Reference Dataset*) is constructed around a
post-cutoff evaluation boundary of **2021-09-30**. Reference structures were
initially released after that date. The 7 All-D targets were obtained from a
post-cutoff candidate screen followed by manual review; the 9 Local-D targets
are the locked post-cutoff Local-D candidates.

### All-D targets

| Target | PDB | Entity | Task |
|---|---|---:|---|
| `7P5R_2` | 7P5R | 2 | all-D racemic crystallography |
| `7QDI_1` | 7QDI | 1 | all-D de novo/mirror image |
| `8AVR_2` | 8AVR | 2 | all-D racemic crystallography |
| `8BFD_2` | 8BFD | 2 | all-D racemic crystallography |
| `8GQP_1` | 8GQP | 1 | heterochiral D-binder |
| `8JH6_1` | 8JH6 | 1 | all-D racemic crystallography |
| `8P6Q_2` | 8P6Q | 2 | all-D racemic crystallography |

The authoritative case metadata, including the curated sequence and the
chain-context policy, is `configs/at_af_7.json`. `configs/target_chirality/`
contains the per-case correction/evaluation configs actually used by the
pipeline.

### Local-D post-cutoff targets

| Target | PDB | Entity |
|---|---|---:|
| `7OOJ_1` | 7OOJ | 1 |
| `7P0A_3` | 7P0A | 3 |
| `7P0T_3` | 7P0T | 3 |
| `7W40_6` | 7W40 | 6 |
| `8HR3_1` | 8HR3 | 1 |
| `8HR4_1` | 8HR4 | 1 |
| `8U78_1` | 8U78 | 1 |
| `8UM9_1` | 8UM9 | 1 |
| `9BIC_1` | 9BIC | 1 |

The target residues and their chain/label identities are in
`data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv`.
The complex components used to construct AF3 inputs are in
`22_complex_component_manifest.csv`; the AF3 residue index mapping is in
`25_residue_identifier_mapping.csv`.

## Curation trail

`data/local_d/` contains the source candidate/review tables used by the Local-D
curation pipeline. `scripts/local_d_curated_v1/` builds the v1 case/residue
manifests; `scripts/local_d_curated_v1_1/` freezes the roles, reconstructs the
complex components, checks the cutoff assignment and emits the post-cutoff
manifests. The checked-in v1.1 files that are required to reproduce the
post-cutoff arm are:

```
06_local_d_case_manifest.csv
07_local_d_residue_manifest.csv
14_correction_target_manifest.csv
22_complex_component_manifest.csv
25_residue_identifier_mapping.csv
```

`19_external_resource_manifest.csv` records the RCSB URLs, download times,
local cache paths and SHA-256 digests of the source mmCIF/entity JSON resources.
`20_software_environment.json` records the curation environment. Those files
are provenance records; the post-cutoff runner itself does not access the
network.

## Seeds and inference controls

`configs/inference_seeds_master42.json` is the authoritative seed file. It was
generated with

```python
import random
random.Random(42).sample(range(1, 2**31 - 1), 30)
```

and contains exactly 30 unique seeds. `data/inference_seeds_master42.csv` is
the flat representation. Each target uses the same 30 seeds in every condition.
No seed is selected or discarded after generation.

Generated AF3 inputs use empty MSA and template fields and are run with
`--norun_data_pipeline`. The AF3 parameters are frozen; no fitting, adapter
training or checkpoint selection is performed.

## Reference coordinates

`data/gt_reference/all_d/*.cif` and `data/gt_reference/local_d/*.cif` are the
deposited mmCIF entries used for evaluation. They are read only by the GT
validation and post-hoc metric stages. Neither `generate_af3_inputs.py`,
`run_full_chirality_correction.py` nor the AT prompt compiler accepts a
reference-coordinate argument. Every target config sets
`"reference_coordinates_allowed": false`, and `load_target_config()` rejects a
config that sets it to true.

The reference files and the derived canonical PDBs are distributed for
scientific reproducibility. The PDB entries remain subject to the RCSB PDB
terms of use; see `data/README.md`.

## Generated artefacts that are not tracked

The release deliberately excludes multi-gigabyte AF3 outputs, logs and report
trees. They are reproducible from the scripts, manifests and references:

```
runs_master42_n30/
runs/local_d_curated_v1_1_post_cutoff/
logs_master42_n30/
reports_master42_n30_ccdfix/
results/
manifests_master42_n30/
inputs/
```

The seed-level and case-level CSVs written by the scripts are the intended
artefacts for downstream tables. The raw generated structures and logs are not
required to review the code path, and are ignored by `.gitignore`.
