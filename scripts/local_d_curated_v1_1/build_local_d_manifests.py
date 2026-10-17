from local_d_curated_v1_1_pipeline import main
if __name__ == "__main__":
    raise SystemExit(main(["reconstruct"] + __import__("sys").argv[1:]))
