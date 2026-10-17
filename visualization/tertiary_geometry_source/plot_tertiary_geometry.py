#!/usr/bin/env python3
"""Reproduce the four-panel WGE-3 figure from the supplied target-level tables.

Requirements: Python 3, matplotlib, numpy. No external data are fetched.
Reported mean distances and deviations are preserved as strings for exact labels.
Run: python plot_tertiary_geometry.py --output-dir ./tertiary_geometry
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def rows(text, all_d_count):
    result = []
    for i, line in enumerate(text.strip().splitlines()):
        target, gt, af3, at_af3, af3_dev, at_af3_dev = line.split()
        result.append(dict(target=target, task='All-D' if i < all_d_count else 'Local-D',
                           gt_mean=gt, af3_mean=af3, at_af3_mean=at_af3,
                           af3_deviation=af3_dev, at_af3_deviation=at_af3_dev))
    return result


DATA = [
    dict(key='disulfide', title='Disulfide linkages', panel='a', rows=rows('''
8P6Q_2 2.0438 2.1881 2.0564 7.06 0.62
7P0A_3 2.0368 2.0589 2.0741 1.09 1.83
7P0T_3 2.0435 2.1002 2.0764 2.77 1.61
7W40_6 2.0473 2.0683 1.9810 1.03 3.24
8HR3_1 2.0313 2.0314 2.0397 0.005 0.41
8HR4_1 2.0307 2.0104 2.0477 1.00 0.84
''', 1)),
    dict(key='hydrogen_bond', title='Hydrogen bonds', panel='b', rows=rows('''
7P5R_2 3.0164 3.0342 3.0121 0.59 0.14
7QDI_1 3.0404 2.9377 3.1167 3.38 2.51
8AVR_2 3.0075 3.0586 3.0188 1.70 0.38
8BFD_2 3.1016 2.9911 3.0025 3.56 3.20
8GQP_1 2.9902 3.0151 2.9680 0.83 0.74
8JH6_1 3.0246 3.0079 3.0369 0.55 0.41
8P6Q_2 2.9572 3.0071 3.1072 1.69 5.07
7OOJ_1 3.0271 2.9518 2.9467 2.49 2.66
7P0A_3 2.9956 2.9619 2.9721 1.12 0.78
7P0T_3 2.9817 2.9604 2.9699 0.71 0.40
7W40_6 3.0234 2.9629 2.9872 2.00 1.20
8HR3_1 3.3650 3.0079 2.9587 10.61 12.07
8HR4_1 3.0420 2.9734 3.0220 2.26 0.66
8U78_1 3.0668 2.8903 2.9065 5.76 5.23
8UM9_1 2.8516 2.9641 2.9610 3.95 3.84
9BIC_1 3.0191 2.9876 2.9007 1.04 3.92
''', 7)),
    dict(key='hydrophobic', title='Hydrophobic contacts', panel='c', rows=rows('''
7P5R_2 3.3940 3.4028 3.2554 0.26 4.08
7QDI_1 3.3304 3.2424 3.2094 2.64 3.63
8AVR_2 3.4703 3.4468 3.2737 0.68 5.67
8BFD_2 3.2830 3.2449 3.1890 1.16 2.86
8GQP_1 3.4305 3.3946 3.3615 1.05 2.01
8JH6_1 3.3886 3.4242 3.3910 1.05 0.07
8P6Q_2 3.4150 3.4106 3.3933 0.13 0.64
7OOJ_1 3.4613 3.4052 3.4032 1.62 1.68
7P0A_3 3.4648 3.4286 3.4181 1.04 1.35
7P0T_3 3.4566 3.4279 3.4180 0.83 1.12
7W40_6 3.4129 3.4095 3.4100 0.10 0.08
8HR3_1 3.3387 3.2939 3.3175 1.34 0.63
8HR4_1 3.3533 3.3106 3.3225 1.27 0.92
8U78_1 3.5158 3.6760 3.5886 4.56 2.07
8UM9_1 3.4675 3.5086 3.4861 1.19 0.54
9BIC_1 3.4214 3.4254 3.2964 0.12 3.65
''', 7)),
    dict(key='salt_bridge', title='Salt bridges', panel='d', rows=rows('''
8GQP_1 3.2250 3.2348 3.3315 0.30 3.30
8JH6_1 3.2291 3.1116 3.2723 3.64 1.34
8P6Q_2 3.3660 3.3237 3.5467 1.26 5.37
7OOJ_1 3.3569 3.2115 3.1888 4.33 5.01
7P0A_3 3.3148 3.1525 3.2401 4.90 2.25
7P0T_3 3.1835 3.1971 3.2511 0.43 2.12
7W40_6 3.3699 3.1225 3.1577 7.34 6.30
8UM9_1 3.1400 2.8520 3.4066 9.17 8.49
''', 3)),
]

COLORS = dict(gt='#707987', af3='#BCA5C8', af3_text='#725684',
              at_af3='#405581', ink='#303744', rule='#E1E5EE', band='#F6F7FB')


def draw_panel(fig, dataset, left, bottom, height, width=0.456):
    # The row bands, target labels and mean-distance columns share one axis.
    frame = fig.add_axes([left, bottom, width, height])
    frame.set_xlim(0, 1)
    slots = []
    for task in ('All-D', 'Local-D'):
        slots.append((task, None))
        slots.extend((None, r) for r in dataset['rows'] if r['task'] == task)
    n = len(slots)
    frame.set_ylim(n - 0.45, -0.55)
    frame.axis('off')

    chart_left = left + 0.112
    chart_width = 0.157
    bar = fig.add_axes([chart_left, bottom, chart_width, height], facecolor='none')
    bar.set_xlim(0, 16)
    bar.set_ylim(frame.get_ylim())
    bar.set_yticks([])
    bar.set_xticks([0, 5, 10, 15])
    bar.xaxis.tick_top()
    bar.tick_params(axis='x', labelsize=5.8, colors=COLORS['gt'], length=2,
                    width=0.55, pad=2)
    for spine in bar.spines.values():
        spine.set_visible(False)
    bar.spines['top'].set_visible(True)
    bar.spines['top'].set_color(COLORS['rule'])
    bar.spines['top'].set_linewidth(0.65)
    for x in (0, 10, 15):
        bar.axvline(x, color=COLORS['rule'], linewidth=0.5, zorder=0)
    bar.axvline(5, color='#A2AABB', linestyle=(0, (3, 3)), linewidth=0.65, zorder=0)

    within = sum(float(r['at_af3_deviation']) <= 5 for r in dataset['rows'])
    top = bottom + height
    fig.text(left, top + 0.061, dataset['panel'], fontsize=10, fontweight='bold',
             color=COLORS['ink'], ha='left', va='bottom')
    fig.text(left + 0.028, top + 0.061, dataset['title'], fontsize=8.7,
             fontweight='bold', color=COLORS['ink'], ha='left', va='bottom')
    fig.text(left + width, top + 0.061, f'Within 5%: {within}/{len(dataset["rows"])}',
             fontsize=6.1, color=COLORS['gt'], ha='right', va='bottom')
    fig.text(chart_left + chart_width/2, top + 0.034, 'Deviation from GT (%)',
             fontsize=6.4, color=COLORS['ink'], ha='center')
    fig.text(left + 0.363, top + 0.034, 'Mean distance (Å)', fontsize=6.4,
             color=COLORS['ink'], ha='center')
    columns = [(0.307, 'GT', 'gt_mean', COLORS['gt']),
               (0.363, 'AF3', 'af3_mean', COLORS['af3_text']),
               (0.419, 'AT-AF3', 'at_af3_mean', COLORS['at_af3'])]
    for x, name, _, color in columns:
        fig.text(left + x, top + 0.010, name, fontsize=6.4, fontweight='bold',
                 color=color, ha='center', va='bottom')

    band_index = 0
    for y, (task, row) in enumerate(slots):
        if task:
            frame.axhline(y + 0.36, color=COLORS['rule'], linewidth=0.6,
                          xmin=0, xmax=1)
            frame.text(0.015, y, task, fontsize=6.4, fontstyle='italic',
                       color=COLORS['gt'], va='center', ha='left')
            continue
        if band_index % 2 == 0:
            frame.axhspan(y - 0.47, y + 0.47, color=COLORS['band'], zorder=0)
        band_index += 1
        frame.text(0.223, y, row['target'], fontsize=6.5,
                   color=COLORS['ink'], ha='right', va='center')
        for x, _, key, color in columns:
            frame.text(x/width, y, row[key], fontsize=6.1, color=color,
                       va='center', ha='center')
        for offset, key, color, text_color in (
            (-0.215, 'af3_deviation', COLORS['af3'], COLORS['af3_text']),
            (+0.215, 'at_af3_deviation', COLORS['at_af3'], COLORS['at_af3']),
        ):
            value = float(row[key])
            bar.barh(y + offset, value, height=0.20, color=color, edgecolor='none',
                     zorder=3)
            # A point also makes very small values visible at their true position.
            bar.plot(value, y + offset, 'o', markersize=1.75, color=color,
                     markeredgewidth=0, zorder=4)
            bar.text(value + 0.30, y + offset, row[key], fontsize=5.55,
                     color=text_color, ha='left', va='center', clip_on=False, zorder=5)
    # A quiet separator distinguishes the bar scale from the numeric means.
    frame.axvline(0.285/width, color=COLORS['rule'], linewidth=0.6)
    return bar, frame


CAPTION = r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{tertiary_geometry.pdf}
\caption{\textbf{AT-AF3 retains close agreement with reference mean interaction distances.}
\textbf{a}, Disulfide linkages. \textbf{b}, Hydrogen bonds.
\textbf{c}, Hydrophobic contacts. \textbf{d}, Salt bridges.
Paired bars show the reported absolute relative deviations of AF3 (lavender)
and AT-AF3 (deep blue) from experimental ground truth (GT); shorter bars indicate
closer agreement. The aligned columns report target-level mean distances
in angstroms for GT, AF3 and AT-AF3. All-D and Local-D targets are grouped
within each panel. The dashed line marks a descriptive 5\% guide,
not a statistical significance or physical validity threshold.
Panel counts indicate the number of AT-AF3 target means within 5\% of GT.
All 46 reported target--interaction comparisons are shown.
AT-AF3 denotes the complete workflow, including geometric correction.
The figure plots target means without uncertainty estimates.}
\label{fig:tertiary_geometry}
\end{figure*}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('tertiary_geometry'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert [len(d['rows']) for d in DATA] == [6, 16, 16, 8]
    counts = [sum(float(r['at_af3_deviation']) <= 5 for r in d['rows']) for d in DATA]
    assert counts == [6, 13, 15, 4]
    assert sum(counts) == 38
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.unicode_minus': False,
    })
    # 180 x 200 mm at full size; no resampling is needed for the vector exports.
    fig = plt.figure(figsize=(180/25.4, 200/25.4))
    positions = [(0.025, 0.547), (0.535, 0.547),
                 (0.025, 0.088), (0.535, 0.088)]
    for dataset, (left, bottom) in zip(DATA, positions):
        draw_panel(fig, dataset, left, bottom, height=0.331)
    handles = [
        Line2D([0], [0], color=COLORS['af3'], lw=3, label='AF3'),
        Line2D([0], [0], color=COLORS['at_af3'], lw=3,
               label='AT-AF3 (including geometric correction)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.51, 0.992),
               ncol=2, frameon=False, fontsize=7.2, handlelength=1.6,
               columnspacing=2.5, borderaxespad=0)
    fig.text(0.5, 0.043, 'AT-AF3 within 5% of GT: 38/46 reported comparisons',
             ha='center', fontsize=7.0, color=COLORS['ink'])
    fig.text(0.5, 0.024, 'Shorter bars indicate closer agreement. Dashed line: 5% descriptive guide.',
             ha='center', fontsize=6.2, color=COLORS['gt'])
    metadata = dict(Title='Tertiary interaction geometry of GT, AF3 and AT-AF3',
                    Subject='Target-level means and reported relative deviations; four interaction types')
    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 600} if ext == 'png' else {}
        if ext == 'pdf':
            kwargs['metadata'] = metadata
        buffer = io.BytesIO()
        fig.savefig(buffer, format=ext, **kwargs)
        payload = buffer.getvalue()
        if ext == 'pdf':
            assert payload.rstrip().endswith(b'%%EOF'), 'PDF export did not finish.'
        destination = out / f'tertiary_geometry.{ext}'
        destination.write_bytes(payload)
        assert destination.read_bytes() == payload, 'Export write was incomplete.'
    plt.close(fig)
    (out/'tertiary_geometry_data.json').write_text(json.dumps(DATA, indent=2)+'\n')
    (out/'tertiary_geometry_caption.tex').write_text(CAPTION)
    print(json.dumps({'output_directory': str(out.resolve()),
                      'reported_comparisons': 46, 'within_5_percent': counts,
                      'png_dpi': 600, 'size_mm': [180, 200]}))


if __name__ == '__main__':
    main()
