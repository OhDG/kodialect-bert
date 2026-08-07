import argparse
import csv
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


STOP_REQUESTED = False


def request_stop(*_args) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def query_gpu(gpu_id: int):
    command = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 6:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout}")
    return values


def monitor(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "gpu_name",
        "gpu_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "temperature_c",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        f.flush()
        while not STOP_REQUESTED:
            try:
                values = query_gpu(args.gpu_id)
                writer.writerow([datetime.now().isoformat(timespec="seconds"), *values])
                f.flush()
            except Exception as exc:
                writer.writerow([datetime.now().isoformat(timespec="seconds"), f"ERROR: {exc}"])
                f.flush()
            time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample NVIDIA GPU utilization to CSV.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    monitor(parse_args())

