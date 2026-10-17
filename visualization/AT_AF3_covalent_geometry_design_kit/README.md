# A redesign for the AT-AF3 covalent-geometry figure

Use a compact two-row arrangement: a local covalent-connectivity schematic and
the correction-change distribution above two aligned All-D / Local-D comparisons.
The second attachment is the visual reference. Only two image files were supplied.

This package is a design template, not a recomputed scientific result.
The preview contains no bond-level points or prediction-level distribution.
Its printed summaries were transcribed from the first attachment:

| Measurement | AF3 | Corrected AT-AF3 |
| --- | ---: | ---: |
| All-D overall mean normalized deviation from GT | 0.0406% | 0.9124% |
| Local-D overall mean normalized deviation from GT | 0.1779% | 0.5488% |

The correction-step mean is 0.00049%, and its printed median is 0.00000%.
No measurements were recovered by guessing plotted positions.

## Design decisions

1. Align All-D and Local-D horizontally, sharing the 11-row order, row positions,
   and x-axis range. This makes the two tasks directly comparable.
2. Keep the dumbbell charts. Their paired endpoints already communicate the
   model comparison efficiently. Use small hollow grey-blue AF3 markers,
   solid navy Corrected AT-AF3 markers, and thin neutral connectors.
3. Group the 11 classes into five bond families using faint background bands.
   Shared row labels appear once, at the left. Light vertical guides help read
   values; avoid a heavy grid or enclosing chart borders.
4. Replace the vertically scattered correction points with an empirical
   cumulative distribution function (ECDF). Its x-axis is normalized bond-length
   change, and its y-axis is cumulative percentage of predictions. Repeated
   zeros become a visible vertical jump, rather than a pile of overlapping
   circles. Every supplied row contributes; the script does not thin or sample.
5. Draw a small connectivity schematic with conventional atom labels and five
   restrained bond colours. The atom layout is a schematic, not a measured
   molecular conformation, and makes no claim about stereochemical orientation.
6. Put summary values beneath the plots as plain text. Remove the rounded
   annotation boxes from the supplied figure.

## Palette and typography

| Role | Colour |
| --- | --- |
| Corrected AT-AF3 | Navy `#354872` |
| AF3 | Grey-blue `#9CAAC5`, hollow marker |
| Secondary schematic accent | Lavender `#B28DB8` |
| Text | Charcoal `#303645` |
| Guides | Light grey `#E3E7EF` |

The default canvas is 7.1 by 5.45 inches, approximately 180 by 138 mm.
It uses Arial where available, with Liberation Sans and DejaVu Sans fallbacks.
Axis text is about 7–8 pt at that physical size, and panel letters are 10.5 pt.
Inspect at the intended final print size. Avoid shrinking the figure to one
column without enlarging text or simplifying the layout.

## Python workflow

Requirements: Python, NumPy, and Matplotlib.

```bash
python -m pip install numpy matplotlib
python redesign_covalent_geometry.py --preview --out AT_AF3_covalent_geometry_layout_preview --dpi 300
```

For measured output, fill the numeric cells in `bond_classes_template.csv` and
save as `bond_classes.csv`. Each task requires one measured value for each of
the 11 listed classes. Keep the task strings and machine-readable bond IDs.

Fill `correction_changes_template.csv` with the underlying observations and
save as `correction_changes.csv`. Only `change_pct` is required; the other columns
retain provenance. Use the same statistical unit and normalization as the
original analysis. A value of `0.01` means **0.01%**, not 1%.
The plotting code preserves signed changes if your operational metric is signed;
it does not convert them to absolute values or infer a metric definition.

```bash
python redesign_covalent_geometry.py \
  --bonds bond_classes.csv \
  --changes correction_changes.csv \
  --summary overall_summary_from_attached_figure.csv \
  --out covalent_geometry
```

This creates an SVG with text retained as text, and a 600-dpi PNG.
The PNG and SVG use the same layout. Values outside a requested `--xmax` are
rejected instead of silently clipped. Both task panels always use the same limit.

The summary CSV is optional. Its supplied values belong to the attached figure;
replace them if you use a different dataset or aggregation. The script never
computes an overall mean by equally averaging the 11 class summaries, because
that could change the original weighting. Preserve the common evaluable domain,
residue context, and aggregation used in your analysis.

## PowerPoint 2019 workflow

1. Use a blank slide with a custom size of approximately 18 by 13.8 cm.
2. Insert `AT_AF3_covalent_geometry_layout_preview.svg` through Insert > Pictures to try the layout.
   After running on measured CSVs, insert the measured SVG instead.
3. PowerPoint 2019 for Windows supports conversion of SVG graphics into Office
   shapes. Right-click the SVG and use Convert to Shape, then ungroup if needed.
   This can help edit lines and colours. Complex SVG text can import differently;
   for substantial text edits, replace labels with native PowerPoint text boxes.
4. Use Align and Distribute to keep the two task charts level. Keep all row
   positions and x-axis limits synchronized.
5. Export the final slide to PDF for a vector manuscript figure, and inspect
   the output at publication size. Use the generated PNG for raster previews.

Python is best for repeatable measured charts, while PowerPoint is convenient
for the final composition. A PowerPoint-only version is possible, but plotting
the 22 paired measurements manually makes updates and scale consistency harder.

## JavaScript alternative

Use D3 with an SVG canvas: one shared linear x-scale for the two task comparisons,
a shared categorical y-scale for the 11 rows, and circles/lines for paired values.
Draw the ECDF as a step path. Keep labels as SVG text, then save the SVG for use
in PowerPoint. JavaScript is useful if you also want an interactive version;
Python is the more direct workflow for this static manuscript figure.

## Scientific interpretation

Deviation from GT and change during the correction step compare different pairs:
AF3 versus Corrected AT-AF3 relative to GT in the task panels, and AT-AF3 versus
Corrected AT-AF3 in the correction panel. A small correction-step change does not
mean that every bond class has a smaller deviation from GT.
Do not title the figure "improved bond geometry": the supplied overall deviation
values are higher for Corrected AT-AF3 than for AF3. A neutral title such as
"Covalent geometry after stereochemical control" accurately frames the figure.
Report the normalized mean deviations and the correction-stage change explicitly.

## Documentation

- Microsoft, SVG editing: https://support.microsoft.com/en-us/office/graphics-visuals/edit-svg-images-in-microsoft-365
- Microsoft, PowerPoint 2019 for Windows: https://support.microsoft.com/en-us/powerpoint/what-s-new-in-powerpoint-2019-for-windows
- Matplotlib, custom styling: https://matplotlib.org/stable/users/explain/customizing.html
- Matplotlib, SVG font handling: https://matplotlib.org/stable/users/explain/text/fonts.html
