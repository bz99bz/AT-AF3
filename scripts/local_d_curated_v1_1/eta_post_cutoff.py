from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "local_d_curated_v1_1_post_cutoff"
GEN_STATUS = RUN_ROOT / "status" / "post_cutoff_generation_status.csv"
CORR_STATUS = RUN_ROOT / "status" / "post_cutoff_correction_status.csv"


def parse_time(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(label: str, rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("status") in {"success", "failed"}]
    running = [r for r in rows if r.get("status") == "running"]
    durations = []
    for r in done:
        start = parse_time(r.get("start_time", ""))
        end = parse_time(r.get("end_time", ""))
        if start and end and end >= start:
            durations.append((end - start).total_seconds())
    avg = sum(durations) / len(durations) if durations else 0.0
    remaining = len(rows) - len(done)
    eta = timedelta(seconds=int(avg * remaining)) if avg > 0 else "unknown"
    return {
        "label": label,
        "finished": len(done),
        "total": len(rows),
        "running": len(running),
        "remaining": remaining,
        "avg_sec_per_task": avg,
        "eta": eta,
    }


def print_summary(s: dict) -> None:
    pct = 100.0 * s["finished"] / s["total"] if s["total"] else 100.0
    print(f"{s['label']}: finished={s['finished']}/{s['total']} ({pct:.1f}%)")
    print(f"{s['label']}: running={s['running']}")
    print(f"{s['label']}: remaining={s['remaining']}")
    print(f"{s['label']}: avg_sec_per_task={s['avg_sec_per_task']:.1f}")
    print(f"{s['label']}: eta={s['eta']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generation", "correction", "all"], default="all")
    args = parser.parse_args(argv)
    summaries = []
    if args.phase in {"generation", "all"}:
        summaries.append(summarize("generation", read_rows(GEN_STATUS)))
    if args.phase in {"correction", "all"}:
        summaries.append(summarize("correction", read_rows(CORR_STATUS)))
    for i, s in enumerate(summaries):
        if i:
            print("---")
        print_summary(s)
    if args.phase == "all" and len(summaries) == 2:
        rows = read_rows(GEN_STATUS) + read_rows(CORR_STATUS)
        print("---")
        print_summary(summarize("all", rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
