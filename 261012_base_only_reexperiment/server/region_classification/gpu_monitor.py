import argparse
import csv
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil

try:
    import pynvml
except ImportError as exc:  # pragma: no cover - checked by the runner
    raise RuntimeError("Install nvidia-ml-py to collect process GPU metrics.") from exc


STOP_REQUESTED = False
NVML_NOT_AVAILABLE = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", (1 << 64) - 1)


def request_stop(*_args) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def tracked_process_ids(root_pid: int) -> Set[int]:
    pids = {root_pid}
    try:
        root = psutil.Process(root_pid)
        pids.update(child.pid for child in root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return pids


def _running_processes(handle) -> List[object]:
    processes: Dict[int, object] = {}
    for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
        getter = getattr(pynvml, getter_name, None)
        if getter is None:
            continue
        try:
            for process in getter(handle):
                processes[process.pid] = process
        except pynvml.NVMLError:
            continue
    return list(processes.values())


def query_process_vram_mib(handle, pids: Set[int]) -> Optional[float]:
    total_bytes = 0
    matched = False
    supported = False
    for process in _running_processes(handle):
        if process.pid not in pids:
            continue
        matched = True
        used = getattr(process, "usedGpuMemory", None)
        if used is None or used == NVML_NOT_AVAILABLE or used < 0:
            continue
        supported = True
        total_bytes += int(used)
    if supported:
        return total_bytes / (1024**2)
    if matched:
        return None
    return 0.0


class ProcessUtilizationSampler:
    def __init__(self, handle) -> None:
        self.handle = handle
        self.last_seen_timestamp = int(time.time() * 1_000_000)
        self.supported = hasattr(pynvml, "nvmlDeviceGetProcessUtilization")

    def sample(self, pids: Set[int]) -> Optional[float]:
        if not self.supported:
            return None
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(
                self.handle,
                self.last_seen_timestamp,
            )
        except pynvml.NVMLError_NotFound:
            samples = []
        except pynvml.NVMLError:
            self.supported = False
            return None

        if samples:
            self.last_seen_timestamp = max(sample.timeStamp for sample in samples)
        latest_by_pid: Dict[int, object] = {}
        for sample in samples:
            if sample.pid in pids:
                previous = latest_by_pid.get(sample.pid)
                if previous is None or sample.timeStamp > previous.timeStamp:
                    latest_by_pid[sample.pid] = sample
        if not latest_by_pid:
            return 0.0
        return min(100.0, sum(float(sample.smUtil) for sample in latest_by_pid.values()))


def monitor(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_id)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")
        utilization_sampler = ProcessUtilizationSampler(handle)

        fields = [
            "timestamp",
            "root_pid",
            "tracked_pids",
            "gpu_name",
            "process_vram_mib",
            "process_sm_utilization_percent",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            f.flush()
            while not STOP_REQUESTED:
                pids = tracked_process_ids(args.root_pid)
                process_vram = query_process_vram_mib(handle, pids)
                process_utilization = utilization_sampler.sample(pids)
                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "root_pid": args.root_pid,
                        "tracked_pids": ";".join(str(pid) for pid in sorted(pids)),
                        "gpu_name": gpu_name,
                        "process_vram_mib": "" if process_vram is None else process_vram,
                        "process_sm_utilization_percent": (
                            "" if process_utilization is None else process_utilization
                        ),
                    }
                )
                f.flush()
                time.sleep(args.interval)
    finally:
        pynvml.nvmlShutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample GPU metrics for one root PID and its descendants only."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--root_pid", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    monitor(parse_args())
