#!/usr/bin/env python3
"""Estimate the remaining four-GPU translation queues using only the stdlib.

The old, uncontended RTX A6000 log contains 19,730.6926 seconds of training
(including epoch validation), 15:46 of final validation, and 6,619 of 17,371
test batches after 1:45:06. Linear test extrapolation gives 4.597 hours and
10.341 hours per translation run before startup, metrics, and writing output.
The planning baseline is therefore 10.5--12 hours/run; it assumes no measured
RTX 4090 speedup. Twenty runs on four balanced lanes take 52.5--60 hours,
plus decoder pretraining, cache preparation, and other unmeasured overhead.
"""

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


TOKENIZERS = ["dialect", "klue", "kobert", "mbert"]
SEEDS = [13, 21, 42, 87, 100]
BASELINE_LOW_HOURS = 10.5
BASELINE_HIGH_HOURS = 12.0
DECODER_STAGE = "02_pretrain_shared_decoder"


def default_config():
    return {
        "gpus": [str(index) for index in range(4)],
        "tokenizers": list(TOKENIZERS),
        "seeds": list(SEEDS),
        "assignments": {str(index): [name] for index, name in enumerate(TOKENIZERS)},
    }


def nonnegative_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def parse_timestamp(value, warnings, name):
    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        warnings.append(f"{name}: invalid timestamp; using recorded elapsed time if available.")
        return None
    if stamp.tzinfo is None:
        warnings.append(f"{name}: timestamp has no timezone; assuming UTC.")
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def elapsed_seconds(stage, name, now, warnings):
    recorded = None
    if stage.get("elapsed_seconds") is not None:
        try:
            recorded = nonnegative_number(stage["elapsed_seconds"], f"{name}.elapsed_seconds")
        except ValueError as exc:
            warnings.append(str(exc) + "; ignoring it.")
    started = parse_timestamp(stage.get("started_at"), warnings, f"{name}.started_at")
    ended = (
        now
        if stage.get("status") == "running"
        else parse_timestamp(stage.get("ended_at"), warnings, f"{name}.ended_at")
    )
    timestamp_elapsed = None
    if started is not None and ended is not None:
        if ended < started:
            warnings.append(f"{name}: end/current time precedes start; ignoring timestamp duration.")
        else:
            timestamp_elapsed = (ended - started).total_seconds()
    if stage.get("status") == "running":
        candidates = [value for value in (recorded, timestamp_elapsed) if value is not None]
        return max(candidates) if candidates else None
    return recorded if recorded is not None else timestamp_elapsed


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("run_config.json must contain an object")
    gpus = [str(gpu) for gpu in config.get("gpus", [])]
    tokenizers = config.get("tokenizers", [])
    seeds = config.get("seeds", [])
    assignments = config.get("assignments", {})
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("gpus must be a nonempty list of unique GPU identifiers")
    if not isinstance(tokenizers, list) or not tokenizers or any(
        not isinstance(name, str) or not name for name in tokenizers
    ) or len(tokenizers) != len(set(tokenizers)):
        raise ValueError("tokenizers must be a nonempty list of unique names")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of unique integers")
    if not isinstance(assignments, dict) or set(assignments) != set(gpus):
        raise ValueError("assignments must contain exactly one entry for each configured GPU")
    assigned = []
    for gpu in gpus:
        if not isinstance(assignments[gpu], list):
            raise ValueError(f"assignments[{gpu!r}] must be a list of tokenizer names")
        assigned.extend(assignments[gpu])
    if sorted(assigned) != sorted(tokenizers):
        raise ValueError("assignments must assign every tokenizer exactly once")
    return gpus, tokenizers, seeds, assignments


def build_estimate(config=None, state=None, decoder_hours=0.0, now=None):
    """Return a JSON-compatible rough projection, never a completion guarantee.

    decoder_hours is a caller-supplied *remaining* decoder budget. It is not
    reduced by elapsed time, and is ignored when the decoder is completed.
    Running translation elapsed time is deducted with a zero floor; an overrun
    adds an explicit unknown remainder instead of implying the job is finished.
    """
    config = default_config() if config is None else config
    state = {"stages": {}} if state is None else state
    decoder_hours = nonnegative_number(decoder_hours, "decoder_hours")
    gpus, tokenizers, seeds, assignments = validate_config(config)
    if not isinstance(state, dict) or not isinstance(state.get("stages", {}), dict):
        raise ValueError("state.json must contain a stages object")
    stages = state.get("stages", {})
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    warnings = []
    durations = {tokenizer: [] for tokenizer in tokenizers}
    jobs = {}
    for tokenizer in tokenizers:
        for seed in seeds:
            name = f"03_translate_{tokenizer}_seed_{seed}"
            stage = stages.get(name, {})
            if not isinstance(stage, dict):
                raise ValueError(f"{name}: stage state must be an object")
            status = stage.get("status", "pending")
            if status not in {"pending", "running", "completed", "failed"}:
                raise ValueError(f"{name}: unsupported stage status {status!r}")
            elapsed = elapsed_seconds(stage, name, now, warnings) if stage else None
            jobs[name] = (stage, status, elapsed)
            if status == "completed":
                if stage.get("timing_incomplete"):
                    warnings.append(
                        f"{name}: completed, but wall time is unknown because an interrupted "
                        "attempt has incomplete timing; excluded from calibration."
                    )
                elif elapsed is not None and elapsed > 0:
                    durations[tokenizer].append(elapsed / 3600.0)
                else:
                    warnings.append(f"{name}: completed without a positive duration; excluded from calibration.")
    pooled = [hours for values in durations.values() for hours in values]
    calibrations = {}
    for tokenizer in tokenizers:
        values = durations[tokenizer] or pooled
        if values:
            mean = statistics.mean(values)
            calibrations[tokenizer] = {
                "source": "tokenizer_completed_mean" if durations[tokenizer] else "pooled_completed_mean",
                "sample_count": len(values),
                "hours_low": mean,
                "hours_high": mean,
            }
        else:
            calibrations[tokenizer] = {
                "source": "old_server_planning_baseline",
                "sample_count": 0,
                "hours_low": BASELINE_LOW_HOURS,
                "hours_high": BASELINE_HIGH_HOURS,
            }
    lanes = []
    unknown_running = False
    for gpu in gpus:
        lane = {"gpu": gpu, "tokenizers": assignments[gpu], "jobs": [],
                "completed_runs": 0, "remaining_runs": 0,
                "hours_low": 0.0, "hours_high": 0.0}
        running_count = 0
        for tokenizer in assignments[gpu]:
            estimate = calibrations[tokenizer]
            for seed in seeds:
                name = f"03_translate_{tokenizer}_seed_{seed}"
                stage, status, elapsed = jobs[name]
                if stage.get("gpu") is not None and str(stage["gpu"]) != gpu:
                    warnings.append(f"{name}: recorded GPU differs from assignment; projection uses GPU {gpu}.")
                low, high = estimate["hours_low"], estimate["hours_high"]
                elapsed_hours = elapsed / 3600.0 if elapsed is not None else None
                overrun = False
                if status == "completed":
                    low = high = 0.0
                    lane["completed_runs"] += 1
                else:
                    lane["remaining_runs"] += 1
                    if status == "running":
                        running_count += 1
                        if stage.get("prior_elapsed_seconds"):
                            warnings.append(
                                f"{name}: resumed run; prior attempt time is not deducted because "
                                "training or evaluation may repeat. Only current attempt elapsed time is deducted."
                            )
                        if elapsed_hours is None:
                            warnings.append(f"{name}: no usable elapsed time; budgeting the full run.")
                        else:
                            overrun = elapsed_hours >= high
                            low = max(0.0, low - elapsed_hours)
                            high = max(0.0, high - elapsed_hours)
                            if overrun:
                                unknown_running = True
                                warnings.append(
                                    f"{name}: running for {elapsed_hours:.2f} h, at or beyond its expected "
                                    "duration; additional running time is unknown, not zero."
                                )
                    elif status == "failed":
                        warnings.append(
                            f"{name}: failed; budgeting a full run. Resume may repeat evaluation "
                            "or require retraining, so the actual remainder is uncertain."
                        )
                lane["hours_low"] += low
                lane["hours_high"] += high
                lane["jobs"].append({"stage": name, "tokenizer": tokenizer, "seed": seed,
                                     "status": status, "elapsed_hours": elapsed_hours,
                                     "hours_low": low, "hours_high": high,
                                     "additional_running_time_unknown": overrun})
        if running_count > 1:
            warnings.append(f"GPU {gpu}: multiple stages marked running; state may be stale and queue ETA unreliable.")
        lanes.append(lane)
    decoder = stages.get(DECODER_STAGE, {})
    if not isinstance(decoder, dict):
        raise ValueError(f"{DECODER_STAGE}: stage state must be an object")
    decoder_completed = decoder.get("status") == "completed"
    decoder_remaining = 0.0 if decoder_completed else decoder_hours
    decoder_unknown = not decoder_completed and decoder_hours == 0
    if decoder_completed and decoder_hours:
        warnings.append("Decoder is completed; ignoring the supplied remaining decoder budget.")
    if decoder.get("status") == "failed":
        warnings.append("Decoder failed; its resume/retraining time needs a separate remaining decoder budget.")
    translation_low = max(lane["hours_low"] for lane in lanes)
    translation_high = max(lane["hours_high"] for lane in lanes)
    completed = sum(lane["completed_runs"] for lane in lanes)
    return {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "completed_translation_runs": completed,
        "total_translation_runs": len(tokenizers) * len(seeds),
        "translation_hours_low": translation_low,
        "translation_hours_high": translation_high,
        "decoder_remaining_hours_budget": decoder_remaining,
        "decoder_remaining_time_unknown": decoder_unknown,
        "additional_running_time_unknown": unknown_running,
        "remaining_hours_low": translation_low + decoder_remaining,
        "remaining_hours_high": translation_high + decoder_remaining,
        "calibration": calibrations,
        "lanes": lanes,
        "warnings": warnings,
        "notes": [
            "Rough wall-clock projection for sequential jobs on each assigned GPU; slowest lane determines completion.",
            "No RTX 4090 speedup is assumed. Baseline: 10.5-12 h/translation run on the prior uncontended RTX A6000 log.",
            "Observed completed-run means are point estimates, not confidence intervals; tokenizer runtimes can differ.",
            "Unbudgeted decoder time, cache preparation, input loading beyond observed stage timing, final summary, and downtime are excluded.",
            "MLM pretraining is not counted; existing MLM encoders are reused.",
            "Running elapsed time is rough and assumes the recorded process is still active; a stopped runner can leave stale state.",
            "Completed timings include recorded attempts; timing_incomplete stages are excluded because their wall time is unknown.",
            "For running resumes, prior attempt time is not deducted: saved progress may require repeated training or evaluation.",
        ],
    }


def hours_range(low, high):
    return f"{low:.1f} h" if math.isclose(low, high) else f"{low:.1f}-{high:.1f} h"


def print_estimate(result):
    print(f"Translation runs completed: {result['completed_translation_runs']}/{result['total_translation_runs']}")
    print("Rough remaining translation queues: " + hours_range(
        result["translation_hours_low"], result["translation_hours_high"]))
    if result["decoder_remaining_hours_budget"]:
        print(f"Additional remaining decoder budget: {result['decoder_remaining_hours_budget']:.1f} h")
        print("Rough remaining total with that budget: " + hours_range(
            result["remaining_hours_low"], result["remaining_hours_high"]))
    if result["decoder_remaining_time_unknown"]:
        print("Decoder/preparation overhead is unmeasured and excluded; supply --decoder-hours for a remaining decoder budget.")
    if result["additional_running_time_unknown"]:
        print("Overrunning jobs have unknown additional time; add that time to the numeric queue projection above.")
    for lane in result["lanes"]:
        print(f"  GPU {lane['gpu']} ({', '.join(lane['tokenizers']) or 'idle'}): "
              f"{lane['remaining_runs']} remaining, " + hours_range(lane["hours_low"], lane["hours_high"]))
        for job in lane["jobs"]:
            if job["status"] == "running":
                elapsed = "unknown" if job["elapsed_hours"] is None else f"{job['elapsed_hours']:.2f} h"
                print(f"    Running {job['stage']}; rough elapsed {elapsed}")
    for tokenizer, calibration in result["calibration"].items():
        print(f"  {tokenizer}: {calibration['source']} "
              f"({calibration['sample_count']} completed timings), "
              + hours_range(calibration["hours_low"], calibration["hours_high"]) + "/run")
    for warning in result["warnings"]:
        print("WARNING: " + warning)
    for note in result["notes"]:
        print("Note: " + note)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="Run directory containing run_config.json and state.json; omit for the 20-run baseline")
    parser.add_argument("--decoder-hours", type=float, default=0.0,
                        help="Additional REMAINING decoder hours to budget (default: unknown/excluded)")
    parser.add_argument("--json", action="store_true", help="Print the full projection as JSON")
    args = parser.parse_args(argv)
    try:
        config = state = None
        if args.run_dir is not None:
            config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8-sig"))
            state = json.loads((args.run_dir / "state.json").read_text(encoding="utf-8-sig"))
        result = build_estimate(config, state, args.decoder_hours)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print_estimate(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
