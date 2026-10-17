#!/usr/bin/env python3
"""Build the All-D target-D-chirality table from a seed-level metrics CSV.

Also reports the per-condition denominator (n_target_chiral_residues) so that any
remaining difference between the AF3 and AT-AF3 site sets is visible in the same
place as the accuracies.

Usage:
    python scripts/ccdfix_table.py --metrics reports_master42_n30_ccdfix/seed_level_metrics.csv \
        --out reports_master42_n30_ccdfix/table_all_d_chirality.md
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

CASE_ORDER = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
COND_LABEL = {"standard_af3": "AF3", "at_af3": "AT-AF3",
              "at_af3_chirality_corrected": "Corrected ATAF3"}
COND_ORDER = ["standard_af3", "at_af3", "at_af3_chirality_corrected"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--denominator-out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.metrics)
    need = {"case_id", "condition", "target_D_accuracy", "n_target_chiral_residues"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"STOP: metrics file lacks columns {sorted(missing)}")

    lines = []
    lines.append("# All-D target D-chirality accuracy (seed-level, no best-of-seed selection)")
    lines.append("")
    lines.append("Mean $\\pm$ sample standard deviation across 30 inference seeds per protein.")
    lines.append("")

    acc_rows = {}
    den_rows = {}
    for case in CASE_ORDER:
        for cond in COND_ORDER:
            sub = df[(df["case_id"] == case) & (df["condition"] == cond)]
            if sub.empty:
                continue
            vals = [float(v) for v in sub["target_D_accuracy"] if pd.notna(v)]
            dens = sorted({int(v) for v in sub["n_target_chiral_residues"] if pd.notna(v)})
            acc_rows[(case, cond)] = (statistics.mean(vals) if vals else float("nan"),
                                      statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals))
            den_rows[(case, cond)] = (dens[0] if len(dens) == 1 else None, sorted(dens))

    lines.append("## Accuracy table")
    lines.append("")
    lines.append("| Protein | " + " | ".join(COND_LABEL[c] for c in COND_ORDER) + " |")
    lines.append("|---|" + "---|" * len(COND_ORDER))
    for case in CASE_ORDER:
        cells = []
        for cond in COND_ORDER:
            if (case, cond) not in acc_rows:
                cells.append("n/a")
                continue
            m, sd, n = acc_rows[(case, cond)]
            cells.append(f"{m:.4f} $\\pm$ {sd:.4f}")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Denominator check (`n_target_chiral_residues`, QC-only)")
    lines.append("")
    lines.append("| Protein | " + " | ".join(COND_LABEL[c] for c in COND_ORDER) + " | equal? |")
    lines.append("|---|" + "---|" * (len(COND_ORDER) + 1))
    mismatched = []
    for case in CASE_ORDER:
        cells = []
        sets = []
        for cond in COND_ORDER:
            if (case, cond) not in den_rows:
                cells.append("n/a")
                continue
            single, allvals = den_rows[(case, cond)]
            cells.append(str(single) if single is not None else f"varies{sorted(set(allvals))}")
            sets.append(tuple(sorted(set(allvals))))
        equal = "yes" if len(set(sets)) == 1 else "**NO**"
        if equal != "yes":
            mismatched.append(case)
        lines.append(f"| {case} | " + " | ".join(cells) + f" | {equal} |")
    lines.append("")

    lines.append("## LaTeX")
    lines.append("")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Protein} & \\textbf{AF3} & \\textbf{AT-AF3} & \\textbf{Corrected ATAF3} \\\\")
    lines.append("\\midrule")
    for case in CASE_ORDER:
        cells = []
        for cond in COND_ORDER:
            if (case, cond) not in acc_rows:
                cells.append("n/a")
                continue
            m, sd, n = acc_rows[(case, cond)]
            txt = f"${m:.4f} \\pm {sd:.4f}$"
            if cond == "at_af3_chirality_corrected":
                txt = f"$\\mathbf{{{m:.4f}}}$"
            cells.append(txt)
        lines.append(f"{case.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
        lines.append("")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    if mismatched:
        lines.append("## Denominator mismatch")
        lines.append("")
        lines.append(f"Denominators differ between conditions for: {', '.join(mismatched)}.")
        lines.append("The affected cases require a common (paired) site set for a like-for-like delta.")
    else:
        lines.append("## Denominators")
        lines.append("")
        lines.append("All three conditions are evaluated over identical site sets for every case.")

    lines.append("")
    lines.append("## Denominator conventions")
    lines.append("")
    lines.append("Two denominator conventions are reported here and are NOT interchangeable:")
    lines.append("")
    lines.append("* **AF3 and AT-AF3** are scored over the full predefined D-target set.  A site that")
    lines.append("  cannot be evaluated (no resolvable CA, e.g. a residue built as a non-amino-acid)")
    lines.append("  stays in the denominator and is counted as incorrect.")
    lines.append("* **Corrected ATAF3** is scored only over the *evaluable* predefined D-target sites,")
    lines.append("  i.e. the non-evaluable sites are removed from the denominator as well.")
    lines.append("")
    lines.append("Consequence: the AT-AF3 value is a lower bound (it carries the non-evaluable sites")
    lines.append("as automatic failures) while the Corrected value has them removed, so part of the")
    lines.append("AT-AF3 -> Corrected gain comes from the denominator change rather than from a")
    lines.append("stereochemical improvement.  When the non-evaluable sites are caused by a residue")
    lines.append("that could never be corrected, the two columns are not measuring the same site set.")
    lines.append("")
    lines.append("A directly comparable delta restricts both conditions to the sites that are")
    lines.append("evaluable in BOTH conditions.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
