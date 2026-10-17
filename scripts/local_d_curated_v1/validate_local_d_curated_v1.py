"""Validate Local-D curated-v1 outputs and no-leakage invariants."""
from local_d_curated_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["validate"] + __import__("sys").argv[1:]))
