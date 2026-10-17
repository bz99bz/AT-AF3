"""Audit existing Local-D CSV inputs without modifying source files."""
from local_d_curated_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["audit"] + __import__("sys").argv[1:]))
