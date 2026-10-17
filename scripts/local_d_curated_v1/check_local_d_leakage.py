"""Run duplicate/permanent-test leakage checks on Local-D curated-v1."""
from local_d_curated_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["leakage"] + __import__("sys").argv[1:]))
