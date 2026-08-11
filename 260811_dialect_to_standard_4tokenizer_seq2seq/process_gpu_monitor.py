import argparse
import csv
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


STOP_REQUESTED = False


def request_stop(*_args) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_nvidia_smi(command):
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    ).stdout


def parse_number(value: str) -> Optional[float]:
    cleaned = value.strip().replace("MiB", "").replace("W", "")
    if cleaned in {"", "-", "N/A", "[N/A]"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def query_device(gpu_id: int) -> Dict[str, object]:
    output = run_nvidia_smi(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [value.strip() for value in output.strip().split(",")]
    if len(values) != 6:
        raise RuntimeError(f"Unexpected device query output: {output}")
    return {
        "gpu_name": values[0],
        "device_gpu_utilization_percent": parse_number(values[1]),
        "device_memory_used_mib": parse_number(values[2]),
        "device_memory_total_mib": parse_number(values[3]),
        "device_power_draw_w": parse_number(values[4]),
        "device_temperature_c": parse_number(values[5]),
    }


def query_process_memory(pid: int) -> Optional[float]:
    output = run_nvidia_smi(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    total = 0.0
    found = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
        except ValueError:
            continue
        if row_pid != pid:
            continue
        value = parse_number(parts[1])
        if value is not None:
            total += value
            found = True
    return total if found else None


def query_process_utilization(pid: int, gpu_id: int) -> Dict[str, Optional[float]]:
    try:
        output = run_nvidia_smi(
            ["nvidia-smi", "pmon", "-i", str(gpu_id), "-c", "1", "-s", "um"]
        )
    except Exception:
        return {
            "process_sm_utilization_percent": None,
            "process_memory_utilization_percent": None,
        }
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            row_pid = int(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            return {
                "process_sm_utilization_percent": parse_number(parts[3]),
                "process_memory_utilization_percent": parse_number(parts[4]),
            }
    return {
        "process_sm_utilization_percent": None,
        "process_memory_utilization_percent": None,
    }


def monitor(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "target_pid",
        "process_gpu_memory_mib",
        "process_sm_utilization_percent",
        "process_memory_utilization_percent",
        "gpu_name",
        "device_gpu_utilization_percent",
        "device_memory_used_mib",
        "device_memory_total_mib",
        "device_power_draw_w",
        "device_temperature_c",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        while not STOP_REQUESTED:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "target_pid": args.pid,
                "error": "",
            }
            try:
                row.update(query_device(args.gpu_id))
                row["process_gpu_memory_mib"] = query_process_memory(args.pid)
                row.update(query_process_utilization(args.pid, args.gpu_id))
            except Exception as exc:
                row["error"] = str(exc)
            writer.writerow(row)
            f.flush()
            time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample target-process and whole-device NVIDIA GPU metrics."
    )
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    monitor(parse_args())
