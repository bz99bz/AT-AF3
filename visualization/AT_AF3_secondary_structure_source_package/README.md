# Secondary-structure composition across CHORD

This figure uses all 144 numeric percentages from the supplied LaTeX table:
16 targets, three conditions, and three grouped DSSP classes. Target names,
condition labels, task grouping, row order, and the reported percentages are
preserved. The condition labelled AT-AF3 is the complete workflow after
correction, as defined in the supplied caption.

## Figure

- Panel a: aligned stacked compositions for GT, AT-AF3 w/o correction, and
  AT-AF3. Helix is muted blue, beta-strand/sheet is lavender, and coil/other is
  pale grey-blue. Labels appear inside segments large enough to read; all
  original numbers remain in `secondary_structure_source.csv`.
- Panel b: signed composition differences, computed as AT-AF3 minus GT for
  H, E, and C. Differences are percentage points. Every cell is annotated,
  with the same symmetric colour scale from -30 to +30 percentage points.
  Colour represents direction and size, rather than a claim of improvement.
- All-D and Local-D share the same row positions across both panels.
- The most abundant GT class is retained in 15 of 16 targets. This is a
  composition-level observation, not a residue-level secondary-structure
  agreement score or a test of three-dimensional structural equivalence.

Rows with totals of 99.9% or 100.1% remain as reported. Stacked bars use the
reported values directly and an axis extending to 100.2%, so rounding does
not silently change or clip the inputs. Differences use decimal arithmetic
on the reported values. No per-seed distribution or uncertainty is inferred.

## Files

- `AT_AF3_secondary_structure.pdf`: vector manuscript figure with embedded
  TrueType fonts.
- `AT_AF3_secondary_structure.svg`: vector figure retaining editable text.
- `AT_AF3_secondary_structure.png`: 600-dpi raster export.
- `AT_AF3_secondary_structure_preview.png`: smaller preview.
- `secondary_structure_source.csv`: all supplied input values, in wide format.
- `AT_AF3_minus_GT_percentage_points.csv`: all 48 annotated differences.
- `plot_secondary_structure.py`: reproducible Matplotlib code.
- `caption.tex`: suggested LaTeX caption and figure label.
- `figure_data_summary.json`: numeric provenance and checks.

## Reproduce

Dependencies: Python, NumPy, and Matplotlib.

```bash
python plot_secondary_structure.py --data secondary_structure_source.csv
```

Edit the CSV to update the measurements, then rerun. All three model columns
use the same composition scale. If a future difference exceeds 30 percentage
points, increase `--difference-scale` to retain an honest shared colour scale.
The code rejects a scale that would hide data.

## Size, style, and editing

The figure is 180 mm wide and 142 mm high. Standard figure text uses a
sans-serif font at about 6.2-7 pt, with larger bold panel letters. The design
adopts the reference images' restrained blues and lavender, compact alignment,
light guides, and direct numeric annotations.

The PDF is intended for insertion into the manuscript. The SVG can be inserted
into PowerPoint 2019 on Windows and converted to shapes for further layout edits.
Text font substitution can affect the SVG, so inspect after importing. The PNG
preserves the rendered appearance.

The width, minimum raster resolution, and standard font sizes follow Nature
Machine Intelligence's figure-formatting guidance:
https://www.nature.com/natmachintell/submission-guidelines/aip-and-formatting
