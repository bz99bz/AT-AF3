# Data directory

This directory contains the minimal data needed to reproduce the released
analyses. Generated structures, logs and report trees are not stored here.

## Contents

### `inference_seeds_master42.csv`

The 30 matched AF3 inference seeds, in flat form. The JSON file
`configs/inference_seeds_master42.json` is authoritative and includes the
master seed and sampling parameters.

### `gt_reference/`

Original deposited mmCIF files used **only** for evaluation:

* `all_d/<PDB>.cif` - seven All-D / heterochiral references
* `local_d/<PDB>.cif` - nine Local-D references

The files are downloaded from RCSB PDB. They are not used by generation or by
the deterministic correction. They are included so that the canonicalisation,
MolProbity, DSSP, bond-geometry and WGE-3 stages can be rerun without an
additional download step. PDB entries remain subject to the RCSB PDB terms of
use and the original depositors' terms.

### `local_d/`

Source candidate/review tables for the Local-D curation screen. The
`local_d_curated_v1_1/` directory contains the checked-in frozen manifests
needed by `scripts/local_d_curated_v1_1/prepare_post_cutoff_run.py`:

| File | Purpose |
|---|---|
| `06_local_d_case_manifest.csv` | case ids, entities, splits and roles |
| `07_local_d_residue_manifest.csv` | residue-level curation |
| `14_correction_target_manifest.csv` | target D positions and chirality |
| `22_complex_component_manifest.csv` | chains/sequences used in AF3 inputs |
| `25_residue_identifier_mapping.csv` | source-to-AF3 residue index mapping |
| `19_external_resource_manifest.csv` | RCSB URLs, local cache paths and hashes |
| `20_software_environment.json` | curation environment |
| `21_input_and_cache_hashes.csv` | input/cache hashes |
| `23_assembly_selection_audit.csv` | assembly selection audit |
| `24_component_exclusion_audit.csv` | excluded components |

`15_*`, `27_*`, `30_*`-`41_*` files are produced when the full curation pipeline
is rerun and are intentionally not required for the post-cutoff evaluation.

### `ccd_components_cache.json`

CCD chemistry cache used by WGE-3. It contains the component definitions needed
to interpret D/non-standard residues without shipping the full
`components.cif.gz`. The cache records that source filename and the archive's
SHA-256 digest; the archive itself is not redistributed.

### Recomputing digests

To record the exact data snapshot used by a run:

```bash
find data -type f ! -name README.md -exec sha256sum {} + > /tmp/chord_data_sha256.txt
```

## Data boundary

The target reference coordinates are evaluation-only. The generation and
correction code paths do not accept a reference path; the target configs set
`"reference_coordinates_allowed": false` and the config loader enforces it.
