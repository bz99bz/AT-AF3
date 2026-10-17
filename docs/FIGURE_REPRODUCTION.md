# Figure reproduction

The released chirality heatmap is assembled from measured tables and released
source CSVs. No plotted value is transcribed by hand from an image.

## Figure 2 - chirality heatmap and paired effects

```bash
python scripts/plot_fig2_chirality_heatmap.py
```

Inputs:

* All-D means: `reports_master42_n30_ccdfix/seed_level_metrics.csv`
* Local-D means: `reports_master42_n30_ccdfix/locald_case_condition_accuracy.csv`
  or `scripts/locald_evaluable_table.py`
* Target manifest: `data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv`

The output is `Fig2_chirality_heatmap_paired_effects.{png,svg}` at the
repository root.

## Data-to-panel mapping

The released panel has a corresponding machine-readable source:

| Panel | Source |
|---|---|
| Chirality heatmap | `reports_master42_n30_ccdfix/seed_level_metrics.csv` + local-D table |
