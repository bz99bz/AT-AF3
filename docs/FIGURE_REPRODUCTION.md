# Figure reproduction

The manuscript figures are assembled from measured tables and released source
CSVs. No plotted value is transcribed by hand from an image. The source
packages below are the minimal reproducible units.

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

## Three-panel chirality summary

```bash
python visualization/AT_AF3_chirality_redesign.py --output-dir visualization/figures
```

The source values are embedded in `AT_AF3_chirality_redesign.py` and are the
target-level manuscript means. The script does not reconstruct seed
distributions or confidence intervals.

## Covalent geometry

```bash
python visualization/AT_AF3_covalent_geometry_design_kit/redesign_covalent_geometry.py \
  --bonds visualization/AT_AF3_covalent_geometry_design_kit/bond_classes.csv \
  --changes visualization/AT_AF3_covalent_geometry_design_kit/correction_changes.csv \
  --summary visualization/AT_AF3_covalent_geometry_design_kit/overall_summary_measured.csv \
  --out visualization/AT_AF3_covalent_geometry_design_kit/covalent_geometry
```

`bond_classes.csv` contains the 11 Engh–Huber class means for All-D and Local-D;
`correction_changes.csv` contains the correction-step observations. See the
package README for the distinction between "deviation from GT" and
"correction-step change". A small correction-step change does not imply a
smaller deviation from GT.

## Secondary-structure composition

```bash
python visualization/AT_AF3_secondary_structure_source_package/plot_secondary_structure.py \
  --data visualization/AT_AF3_secondary_structure_source_package/secondary_structure_source.csv
```

The package contains all 144 reported percentages, the all-D/Local-D grouping
and the signed AT-AF3-minus-GT differences. It reports composition, not a
residue-level Q3 score. `scripts/secondary_structure_q3.py` computes the Q3
agreement separately.

## Tertiary interaction geometry

```bash
python visualization/tertiary_geometry_source/plot_tertiary_geometry.py \
  --output-dir visualization/tertiary_geometry_source/tertiary_geometry
```

The source JSON retains the reported target-level interaction distances and
deviations for disulfide, hydrogen-bond, hydrophobic and salt-bridge classes.
The figure shows observed geometry, not contact recovery.

## Data-to-panel mapping

Every panel has a corresponding machine-readable source:

| Panel | Source |
|---|---|
| Chirality heatmap | `reports_master42_n30_ccdfix/seed_level_metrics.csv` + local-D table |
| Chirality summary | embedded manuscript target means in `AT_AF3_chirality_redesign.py` |
| Covalent geometry | `bond_classes.csv`, `correction_changes.csv`, `overall_summary_measured.csv` |
| Secondary structure | `secondary_structure_source.csv` |
| Tertiary geometry | `tertiary_geometry_data.json` |
