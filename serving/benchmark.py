"""Empirical latency and throughput benchmark for the churn API and the batch CLI.

Run from the repository root:
    python -m serving.benchmark

What it measures (all on this machine, client and server on localhost):
  1. In-process timing (no HTTP): schema validation, feature preparation, scoring
     and raw predict_proba for one record, to separate model time from HTTP and
     framework overhead.
  2. API: a uvicorn server is started as a subprocess. Warm-up requests are sent
     first and reported separately as cold-start latency (excluded from the
     statistics). Then N timed requests from one sequential client, and N timed
     requests spread over C concurrent clients, each run repeated R times.
     Payloads cycle through the committed data/sample_input.csv rows, so this
     part works on a fresh clone.
  3. Batch: a large CSV built by REPEATING dataset rows (this measures
     throughput, not accuracy). Whole-run time of `python predict_batch.py`
     including interpreter start-up and model loading, and in-process stage
     times (read, validation, feature preparation, prediction, write).

Raw results are written to artifacts/benchmark_results.json.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from predict_batch import EXPECTED_COLUMNS, read_input, validate_rows
from serving.schemas import PredictRequest
from serving.scoring import prepare_features, score_frame
from src.data_pipeline import FEATURES, ID_COLUMN, PROCESSED_DIR, REPO_ROOT

HOST = "127.0.0.1"
SAMPLE_INPUT = REPO_ROOT / "data" / "sample_input.csv"
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "churn_model.joblib"
WORK_DIR = REPO_ROOT / "data" / "benchmark"          # git-ignored scratch files
RESULTS_PATH = REPO_ROOT / "artifacts" / "benchmark_results.json"
LIBRARIES = ["fastapi", "starlette", "uvicorn", "pydantic", "httpx", "scikit-learn", "lightgbm",
             "numpy", "pandas", "joblib"]


# ---------------------------------------------------------------- helpers
def latency_summary(seconds: list[float]) -> dict:
    """Percentiles and spread of a list of latencies, reported in milliseconds."""
    ms = np.asarray(seconds) * 1000
    return {
        "n": int(ms.size),
        "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean()),
        "min_ms": float(ms.min()), "max_ms": float(ms.max()),
    }


def across_repeats(repeats: list[dict]) -> dict:
    """For every numeric metric: median over the repeats, plus min and max as the spread."""
    out = {}
    for key, value in repeats[0].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values = [r[key] for r in repeats]
            out[key] = {"median": float(statistics.median(values)), "min": float(min(values)), "max": float(max(values))}
    return out


def load_payloads() -> list[dict]:
    """Valid API payloads: the committed sample rows (native Python types)."""
    return pd.read_csv(SAMPLE_INPUT, dtype={ID_COLUMN: str}).to_dict(orient="records")


def environment(port: int) -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "libraries": {name: importlib.metadata.version(name) for name in LIBRARIES},
    }
    if sys.platform == "darwin":
        def sysctl(key):
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        info.update({
            "cpu": sysctl("machdep.cpu.brand_string"),
            "physical_cpus": int(sysctl("hw.physicalcpu")),
            "performance_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0),
            "efficiency_cores": int(sysctl("hw.perflevel1.physicalcpu") or 0),
            "ram_gb": round(int(sysctl("hw.memsize")) / 2**30, 1),
            "macos_version": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
        })
        battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        info["power_source"] = battery.split("'")[1] if battery.count("'") >= 2 else "unknown"
    elif Path("/proc/cpuinfo").exists():
        cpuinfo = Path("/proc/cpuinfo").read_text()
        info["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if "model name" in line), "")
        meminfo = Path("/proc/meminfo").read_text().split()
        info["ram_gb"] = round(int(meminfo[meminfo.index("MemTotal:") + 1]) / 2**20, 1)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                    capture_output=True, text=True).stdout.strip())
        info["git_commit"] = commit + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = "unknown"
    info["server"] = {
        "command": f"python -m uvicorn serving.app:app --host {HOST} --port {port} --workers 1",
        "host": HOST, "network": "localhost (client and server on the same machine)", "workers": 1,
        "event_loop": "uvloop" if importlib.util.find_spec("uvloop") else "asyncio",
        "http_parser": "httptools" if importlib.util.find_spec("httptools") else "h11",
        "logging": "uvicorn access log and app INFO log per request, written to a file",
        "failure_switch": "unset (off)",
    }
    return info


# ---------------------------------------------------------------- 1. in-process timing
def time_in_process(payloads: list[dict], n: int, repeats: int) -> dict:
    """Per-record cost of each serving step without HTTP, using the real functions."""
    bundle = joblib.load(ARTIFACT_PATH)
    records = [{f: p[f] for f in FEATURES} for p in payloads]
    frames = [prepare_features([r], bundle)[0] for r in records]  # prepared once, for the scoring-only steps

    def full_request(i):
        request = PredictRequest.model_validate(payloads[i])
        frame, _ = prepare_features([request.model_dump(include=set(FEATURES))], bundle)
        return score_frame(frame, bundle)

    steps = {  # each takes the index of a sample row
        "validation": lambda i: PredictRequest.model_validate(payloads[i]),
        "prepare_features": lambda i: prepare_features([records[i]], bundle),
        "score_frame": lambda i: score_frame(frames[i], bundle),
        "predict_proba_only": lambda i: bundle["pipeline"].predict_proba(frames[i]),
        "full_in_process_request": full_request,
    }
    results = {}
    for name, step in steps.items():
        for i in range(len(payloads)):  # warm-up, not timed
            step(i)
        repeat_stats = []
        for _ in range(repeats):
            times = []
            for k in range(n):
                start = time.perf_counter()
                step(k % len(payloads))
                times.append(time.perf_counter() - start)
            repeat_stats.append(latency_summary(times))
        results[name] = {"repeats": repeat_stats, "across_repeats": across_repeats(repeat_stats)}
    return results


# ---------------------------------------------------------------- 2. API
def free_port(preferred: int) -> int:
    with socket.socket() as s:
        if s.connect_ex((HOST, preferred)) != 0:
            return preferred
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def start_server(port: int, log_path: Path):
    env = {k: v for k, v in os.environ.items() if k not in ("CHURN_API_SIMULATE_FAILURE", "CHURN_MODEL_PATH")}
    log = open(log_path, "w")
    start = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.app:app", "--host", HOST, "--port", str(port),
                             "--workers", "1"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = start + 60
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see {log_path}")
        try:
            if httpx.get(f"http://{HOST}:{port}/health", timeout=1).status_code == 200:
                return proc, log, time.perf_counter() - start
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    proc.terminate()
    raise RuntimeError(f"server not healthy after 60 s; see {log_path}")


def send(client: httpx.Client, payload: dict):
    """One timed request. Returns (seconds, ok)."""
    start = time.perf_counter()
    try:
        response = client.post("/predict", json=payload)
        elapsed = time.perf_counter() - start
        return elapsed, response.status_code == 200 and response.json()["source"] == "model"
    except httpx.HTTPError:
        return time.perf_counter() - start, False


def api_run(base_url: str, payloads: list[dict], n: int, clients: int, tag: str) -> dict:
    """n requests split evenly over `clients` threads, each with its own keep-alive connection."""
    per_client = [n // clients + (1 if i < n % clients else 0) for i in range(clients)]
    latencies, errors, lock = [], [0], threading.Lock()
    barrier = threading.Barrier(clients + 1)

    def worker(index: int):
        with httpx.Client(base_url=base_url, timeout=30) as client:
            client.get("/health")  # open the connection before the clock starts
            barrier.wait()
            for j in range(per_client[index]):
                k = sum(per_client[:index]) + j
                seconds, ok = send(client, {**payloads[k % len(payloads)], "request_id": f"{tag}-{k}"})
                with lock:
                    if ok:
                        latencies.append(seconds)
                    else:
                        errors[0] += 1

    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [pool.submit(worker, i) for i in range(clients)]
        barrier.wait()
        start = time.perf_counter()
        for f in futures:
            f.result()
        wall = time.perf_counter() - start
    return {**latency_summary(latencies), "errors": errors[0], "wall_s": wall, "requests_per_s": n / wall}


def benchmark_api(payloads: list[dict], n: int, concurrency: int, repeats: int, warmup: int, port: int) -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    proc, log, startup_s = start_server(port, WORK_DIR / "server.log")
    base_url = f"http://{HOST}:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            warm = [send(client, {**payloads[i % len(payloads)], "request_id": f"warm-{i}"}) for i in range(warmup)]
        warm_ms = [s * 1000 for s, _ in warm]
        result = {
            "server_startup_to_healthy_s": startup_s,
            "cold_start": {"warmup_requests": warmup, "first_request_ms": warm_ms[0],
                           "warmup_requests_after_first": latency_summary([s for s, _ in warm[1:]]),
                           "warmup_errors": sum(not ok for _, ok in warm),
                           "note": "excluded from the timed statistics"},
        }
        for label, clients in (("sequential_1_client", 1), (f"concurrent_{concurrency}_clients", concurrency)):
            runs = [api_run(base_url, payloads, n, clients, f"{label}-{r}") for r in range(repeats)]
            result[label] = {"clients": clients, "requests_per_run": n, "repeats": runs,
                             "across_repeats": across_repeats(runs)}
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()
    return result


# ---------------------------------------------------------------- 3. batch
def build_batch_file(n_rows: int, path: Path) -> dict:
    """Large input made by repeating dataset rows; IDs get a '-r<k>' suffix to stay unique."""
    splits = [PROCESSED_DIR / f"{name}.csv" for name in ("train", "val", "test")]
    if all(p.exists() for p in splits):
        source = pd.concat([pd.read_csv(p, dtype={ID_COLUMN: str}) for p in splits]).drop(columns="churn")
        source_label = "all 7,043 processed dataset rows (train + validation + test), target column removed"
    else:
        source = pd.read_csv(SAMPLE_INPUT, dtype={ID_COLUMN: str})
        source_label = "the 20 committed sample rows (processed dataset not available)"
    copies = math.ceil(n_rows / len(source))
    big = pd.concat([source.assign(**{ID_COLUMN: source[ID_COLUMN] + f"-r{k}"}) for k in range(copies)]).head(n_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(path, index=False)
    return {"rows": len(big), "source": source_label, "distinct_source_rows": len(source),
            "note": "rows are REPEATED copies; this file measures throughput, not accuracy",
            "file_mb": round(path.stat().st_size / 2**20, 2)}


def benchmark_batch(n_rows: int, repeats: int) -> dict:
    input_path, output_path = WORK_DIR / "batch_input.csv", WORK_DIR / "batch_output.csv"
    info = build_batch_file(n_rows, input_path)
    rows = info["rows"]

    # Whole run as a user would run it: new Python process, imports, model load, read, validate, score, write.
    whole = []
    for _ in range(repeats):
        start = time.perf_counter()
        done = subprocess.run([sys.executable, "predict_batch.py", "--input", str(input_path),
                               "--output", str(output_path)], cwd=REPO_ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - start
        if done.returncode != 0:
            raise RuntimeError(f"predict_batch.py failed:\n{done.stderr}")
        scored = int((pd.read_csv(output_path, usecols=["status"])["status"] == "scored").sum())
        whole.append({"wall_s": wall, "rows_per_s": rows / wall, "rows_scored": scored})

    # Stage times in-process, using the CLI's own functions.
    bundle = joblib.load(ARTIFACT_PATH)
    stages = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        df = read_input(input_path)[EXPECTED_COLUMNS]
        t1 = time.perf_counter()
        valid, _, rejections = validate_rows(df)
        t2 = time.perf_counter()
        frame, _ = prepare_features(valid, bundle)
        t3 = time.perf_counter()
        result = score_frame(frame, bundle)
        t4 = time.perf_counter()
        pd.DataFrame({"p": result.probability}).to_csv(WORK_DIR / "stage_output.csv", index=False)
        t5 = time.perf_counter()
        read_s, validate_s, prepare_s, predict_s, write_s = t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4
        stages.append({
            "read_s": read_s, "validation_s": validate_s, "prepare_features_s": prepare_s,
            "prediction_s": predict_s, "write_s": write_s,
            "scoring_only_s": validate_s + prepare_s + predict_s,
            "scoring_only_rows_per_s": rows / (validate_s + prepare_s + predict_s),
            "validation_rows_per_s": rows / validate_s,
            "prediction_rows_per_s": rows / predict_s,
            "rows_rejected": len(rejections), "source": result.source,
        })
    return {"input": info, "whole_run_including_startup": {"repeats": whole, "across_repeats": across_repeats(whole)},
            "in_process_stages": {"repeats": stages, "across_repeats": across_repeats(stages),
                                  "definitions": {
                                      "scoring_only": "validation + feature preparation + prediction (no file I/O, "
                                                      "no interpreter start-up or model loading)",
                                      "prediction": "score_frame: one vectorised predict_proba call plus threshold"}}}


# ---------------------------------------------------------------- report
def print_summary(results: dict) -> None:
    def fmt(m, digits=2):
        return f"{m['median']:.{digits}f} [{m['min']:.{digits}f}-{m['max']:.{digits}f}]"
    env = results["environment"]
    print(f"\nMachine: {env.get('cpu', '?')}, {env.get('ram_gb', '?')} GB RAM, {env['os']}, Python {env['python']}")
    print("\nIn-process, one record (median of repeats [min-max of repeats]):")
    for name, r in results["in_process"].items():
        a = r["across_repeats"]
        print(f"  {name:25s} p50 {fmt(a['p50_ms'], 3)} ms  p95 {fmt(a['p95_ms'], 3)} ms  "
              f"p99 {fmt(a['p99_ms'], 3)} ms  mean {fmt(a['mean_ms'], 3)} ms")
    api = results["api"]
    print(f"\nAPI server start-up to healthy: {api['server_startup_to_healthy_s']:.2f} s;"
          f" first (cold) request {api['cold_start']['first_request_ms']:.1f} ms")
    for label in [k for k in api if k.startswith(("sequential", "concurrent"))]:
        a = api[label]["across_repeats"]
        print(f"  {label}: p50 {fmt(a['p50_ms'])}  p95 {fmt(a['p95_ms'])}  p99 {fmt(a['p99_ms'])}  "
              f"mean {fmt(a['mean_ms'])}  min {fmt(a['min_ms'])}  max {fmt(a['max_ms'])} ms  "
              f"errors {fmt(a['errors'])}  req/s {fmt(a['requests_per_s'])}")
    b = results["batch"]
    w, s = b["whole_run_including_startup"]["across_repeats"], b["in_process_stages"]["across_repeats"]
    print(f"\nBatch, {b['input']['rows']:,} rows ({b['input']['source']}; repeated rows):")
    print(f"  whole run incl. start-up: {fmt(w['wall_s'])} s -> {fmt(w['rows_per_s'])} rows/s")
    print(f"  validation {fmt(s['validation_s'])} s, prepare {fmt(s['prepare_features_s'])} s, "
          f"prediction {fmt(s['prediction_s'])} s, read {fmt(s['read_s'])} s, write {fmt(s['write_s'])} s")
    print(f"  scoring only {fmt(s['scoring_only_rows_per_s'])} rows/s; validation {fmt(s['validation_rows_per_s'])} "
          f"rows/s; prediction {fmt(s['prediction_rows_per_s'])} rows/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the churn API and batch CLI on this machine.")
    parser.add_argument("--requests", type=int, default=1000, help="timed API requests per run (default 1000)")
    parser.add_argument("--concurrency", type=int, default=8, help="clients in the concurrent run (default 8)")
    parser.add_argument("--repeats", type=int, default=3, help="repeats of every run (default 3)")
    parser.add_argument("--warmup", type=int, default=20, help="untimed warm-up requests (default 20)")
    parser.add_argument("--batch-rows", type=int, default=100_000, help="rows in the batch file (default 100000)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    port = free_port(args.port)
    payloads = load_payloads()
    results = {
        "environment": environment(port),
        "settings": {"requests_per_run": args.requests, "concurrency": args.concurrency, "repeats": args.repeats,
                     "warmup_requests": args.warmup, "batch_rows": args.batch_rows,
                     "payloads": f"{len(payloads)} rows of data/sample_input.csv, cycled; unique request_id each",
                     "client": "httpx, one keep-alive connection per client thread",
                     "percentiles": "numpy.percentile, linear interpolation"},
    }
    print("1/3 in-process timing ...", flush=True)
    results["in_process"] = time_in_process(payloads, args.requests, args.repeats)
    print("2/3 API benchmark ...", flush=True)
    results["api"] = benchmark_api(payloads, args.requests, args.concurrency, args.repeats, args.warmup, port)
    print("3/3 batch benchmark ...", flush=True)
    results["batch"] = benchmark_batch(args.batch_rows, args.repeats)

    args.output.write_text(json.dumps(results, indent=2))
    print_summary(results)
    print(f"\nRaw results saved to {args.output.relative_to(REPO_ROOT) if args.output.is_relative_to(REPO_ROOT) else args.output}")


if __name__ == "__main__":
    main()
