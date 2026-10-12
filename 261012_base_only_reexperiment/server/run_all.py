"""Run both original experiments sequentially on the laboratory server."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate the prepared bundle; do not train.")
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    steps = [
        ["verify_bundle.py"],
        ["region_classification/run_server.py", "--check-only"],
        ["translation/run_server.py", "--check-only"],
    ]
    if not args.check_only:
        steps += [
            ["region_classification/run_server.py"],
            ["region_classification/run_server.py", "--analyze-only"],
            ["translation/run_server.py"],
        ]
    for step in steps:
        command = [sys.executable, str(ROOT / step[0]), *step[1:]]
        print("[RUN] " + " ".join(step), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
