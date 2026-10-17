"""Build Local-D curated-v1 case/residue/manifests from validated inputs."""
from local_d_curated_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["build"] + __import__("sys").argv[1:]))
