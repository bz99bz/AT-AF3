Tertiary interaction geometry figure

The four panels reproduce all reported measurements from the supplied disulfide, hydrogen-bond, hydrophobic-contact and salt-bridge tables. Target order and All-D/Local-D grouping are unchanged. All 230 source numerical strings are retained, including the 0.005% AF3 deviation for 8HR3_1.

Muted lavender bars represent AF3 and deep blue bars represent the complete AT-AF3 workflow, including geometric correction. Cool gray GT, plum AF3 and deep blue AT-AF3 columns give the reported mean distances in angstroms. The palette follows the supplied reference's blue and lavender tones. A shared bar scale enables comparison of relative deviations across the four interaction types. The 5% dashed guide is descriptive. Counts of AT-AF3 target means within 5% are 6/6, 13/16, 15/16 and 4/8, totaling 38/46. The figure uses reported deviations directly rather than recalculating them from rounded means.

GT denotes experimental ground truth. AF3 denotes AlphaFold 3. AT-AF3 denotes the complete Associative Thinking-guided AF3 workflow. The figure displays target-level means; uncertainty estimates and sample-level distributions were not supplied.

Reproduce the figure using Python 3 with matplotlib and numpy installed:
python plot_tertiary_geometry.py --output-dir ./tertiary_geometry

The program exports a 180 x 200 mm vector PDF, an editable SVG with text preserved, and a 600 dpi PNG. It also writes the exact source-data JSON and a LaTeX figure/caption fragment. The PDF embeds its fonts.

For LaTeX, use the graphicx package and place the generated tertiary_geometry.pdf beside the manuscript, then insert the generated tertiary_geometry_caption.tex fragment. The figure label is fig:tertiary_geometry.
