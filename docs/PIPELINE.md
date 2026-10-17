# Pipeline

The release has two evaluation arms and one shared structural-validation stage.
All paths below are relative to the repository root.

## 1. All-D arm

```
configs/inference_seeds_master42.json
configs/at_af_7.json
        │
        ▼
scripts/build_master42_experiment_manifest.py
  manifests_master42_n30/master42_generation_manifest.csv
  manifests_master42_n30/inputs/{standard_af3,at_af3}/<case>/seed_<seed>/fold_input.json
  configs/target_chirality/<case>.json
        │
        ▼
scripts/run_master42_af3_at_af3_batch.py
  runs_master42_n30/standard_af3/<case>/seed_<seed>/
  runs_master42_n30/at_af3/<case>/seed_<seed>/
        │
        ▼
scripts/build_master42_correction_manifest.py
  manifests_master42_n30/master42_correction_manifest.csv
        │
        ▼
scripts/run_master42_chirality_correction_batch.py
  runs_master42_n30/at_af3_chirality_corrected/<case>/seed_<seed>/corrected_all_errors.cif
        │
        ├──────────────► scripts/evaluate_master42_experiment.py
        │                  reports_master42_n30_ccdfix/seed_level_metrics.csv
        │                  reports_master42_n30_ccdfix/case_level_summary.csv
        │                  reports_master42_n30_ccdfix/paired_comparison_summary.csv
        │                  reports_master42_n30_ccdfix/geometry_quality_summary.csv
        │
        └──────────────► scripts/ccdfix_table.py
                           reports_master42_n30_ccdfix/table_all_d_chirality.md
```

`build_master42_experiment_manifest.py` writes a target-chirality config only
when one is not already present. The checked-in configs in
`configs/target_chirality/` are the frozen evaluation definitions and must not be
overwritten: for example, `7P5R_2` designates the D copy on chain C, not the L
enantiomer that the same entry also deposits. The manifest builder therefore
does not replace an existing target config.

The generation manifest is frozen at 7 cases × 30 seeds × 2 conditions = 420
tasks; the correction manifest is 7 cases × 30 seeds = 210 tasks. The batch
scripts assert these counts and refuse to run on a different task set.

### Generation details

`generate_af3_inputs.py` creates both AF3 JSONs:

* **Standard-AF3** contains the standard L sequence and no D-CCD modifications.
* **AT-AF3** contains the same sequence plus residue-level CCD modifications
  compiled by `associative_chirality_memory.py`; glycine is never a target.

Both conditions receive the same `modelSeeds` value. The AF3 input dialect is
`alphafold3` version 2. MSA and template fields are empty, and the model is run
with `--norun_data_pipeline` so that no additional inference information is
used.

### Correction details

`run_full_chirality_correction.py`:

1. loads the predicted structure and the target-chirality config;
2. builds a binary error code for every target site from the signed volume;
3. mirrors only correction-eligible side chains through the local Cα frame;
4. writes `corrected_all_errors.cif`, per-residue correction status, geometry
   diagnostics and a metadata file.

The config declares `"reference_coordinates_allowed": false`; the correction
code checks this field and has no reference-structure argument.

## 2. Local-D post-cutoff arm

```
data/local_d/local_d_curated_v1_1/{06,22,25,14}_*.csv
        │
        ▼
scripts/local_d_curated_v1_1/prepare_post_cutoff_run.py
  runs/local_d_curated_v1_1_post_cutoff/inputs/
  runs/local_d_curated_v1_1_post_cutoff/manifests/
  runs/local_d_curated_v1_1_post_cutoff/status/
        │
        ▼
scripts/local_d_curated_v1_1/run_post_cutoff_linux.py --phase all
  runs/local_d_curated_v1_1_post_cutoff/standard_af3/
  runs/local_d_curated_v1_1_post_cutoff/at_af3_first_pass/
  runs/local_d_curated_v1_1_post_cutoff/corrected/
```

The post-cutoff workflow is 9 cases × 30 seeds × 2 generation conditions = 540
generation tasks plus 270 correction tasks. `prepare_post_cutoff_run.py`
refuses cases outside the nine locked post-cutoff IDs.

The Local-D target manifest is a CSV because the target positions are sparse.
`src/chirality_error_code.py` loads it with the same convention as the All-D
JSON configs: chain id plus label sequence id, with `correction_supported` and
`target_chirality` columns.

## 3. Shared GT validation

```
data/gt_reference/{all_d,local_d}/<PDB>.cif
        │
        ▼
scripts/run_gt_phenix_validation_linux.py
  results/gt_comparison_phenix_ccdfix/canonical_pdb/
  results/gt_comparison_phenix_ccdfix/canonical_chain_id_mapping.json
  results/gt_comparison_phenix_ccdfix/per_structure_molprobity.csv
  results/gt_comparison_phenix_ccdfix/paired_vs_gt.csv
  results/gt_comparison_phenix_ccdfix/per_case_summary.csv
  results/gt_comparison_phenix_ccdfix/gt_phenix_manifest.csv
        │
        ├── scripts/gt_metrics_suite.py
        ├── scripts/tertiary_structure_rmsd.py
        ├── scripts/secondary_structure_q3.py
        ├── scripts/analyze_bond_geometry.py
        ├── scripts/phenix_interaction_analysis.py
        ├── scripts/ccdfix_evaluable_table.py
        └── scripts/locald_evaluable_table.py
```

Canonicalisation is necessary because the deposited entries use author/entity
chain names and may represent D residues as HETATM records. The canonical PDBs
preserve coordinates and chain identity while translating only the labels
required by downstream parsers. The mapping is written to
`canonical_chain_id_mapping.json`.

`run_gt_phenix_validation_linux.py` expects each seed directory to contain one
ranking CSV and six model CIFs per prediction condition, and selects the
highest-ranked sample. This is the AF3 native ranking rule; no seed selection is
performed.

## 4. Table and figure outputs

| Script | Output |
|---|---|
| `ccdfix_table.py` | All-D chirality table from seed-level accuracy |
| `ccdfix_evaluable_table.py` | All-D table restricted to evaluable target sites |
| `locald_evaluable_table.py` | Local-D table from corrected runs and target manifest |
| `metric_tables.py` | One GT/AF3/AT-AF3/corrected table per GT metric |
| `per_case_metric_tables.py` | Per-case tables from `gt_metrics_per_seed.csv` |
| `tertiary_metric_tables.py` | RMSD, radius of gyration and mirror-RMSD tables |
| `plot_fig2_chirality_heatmap.py` | Figure 2 heatmap and paired-effect panels |
| `visualization/AT_AF3_chirality_redesign.py` | Standalone three-panel chirality figure from manuscript values |
| `visualization/AT_AF3_covalent_geometry_design_kit/` | Covalent-geometry figure source data and plotting code |
| `visualization/AT_AF3_secondary_structure_source_package/` | DSSP composition figure and source CSV |
| `visualization/tertiary_geometry_source/` | WGE-3 interaction-geometry figure and source JSON |

