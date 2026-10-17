"""Generate dry-run/future AF3 manifests for Local-D curated-v1.

This intentionally does not launch AF3 and does not materialize incomplete
full-complex JSONs from entity-only source CSV rows.
"""
from local_d_curated_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["inputs"] + __import__("sys").argv[1:]))
