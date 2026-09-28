import os
import sys
import argparse
import subprocess

def main() -> None:
    parser = argparse.ArgumentParser(description="Main entry point for mechanistic interpretability experiments.")
    parser.add_argument(
        "--experiment",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Specify which experiment to run.",
    )
    args = parser.parse_args()
    root_dir = os.getcwd()
    experiment_path = [
        os.path.join(root_dir, "01_induction_heads", "src", "visualize.py"),
    ]

    script = experiment_path[args.experiment - 1]
    print(f"Running experiment {args.experiment} using script: {script}")

    subprocess.run(
        [sys.executable, script],
        cwd=root_dir,
        check=True,
    )
