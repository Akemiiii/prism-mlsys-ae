#!/usr/bin/env python3
"""Sweep max-running-requests and evaluate average throughput across datasets."""

import argparse
import csv
import json
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_DATASETS = [
    "data/mt_bench/question.jsonl",
    "data/humaneval/question.jsonl",
    "data/gsm8k/question.jsonl",
    "data/alpaca/question.jsonl",
    "data/sum/question.jsonl",
    "data/qa/question.jsonl",
]


def wait_for_port(host: str, port: int, timeout: int) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(1)
    return False


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def parse_result_jsonl(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def normalize_algorithm(algorithm: str) -> str:
    a = algorithm.strip().upper()
    return "NONE" if a in {"NONE", "NO", "OFF", "DISABLED"} else a


def normalize_draft_path(path: str) -> str:
    p = path.strip()
    return "" if p in {"", "-", "NONE", "none"} else p


def with_draft_model_prefix(path: str, prefix: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(prefix) / p)


def run_one_dataset(
    python_exe: str,
    bench_script: Path,
    work_dir: Path,
    dataset: str,
    port: int,
    temperature: float,
    num_questions: int,
) -> dict:
    with tempfile.NamedTemporaryFile(prefix="bench_", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        python_exe,
        str(bench_script),
        "--question-file",
        dataset,
        "--temperature",
        str(temperature),
        "--result-file",
        str(tmp_path),
        "--port",
        str(port),
        "--num-questions",
        str(num_questions),
    ]
    proc = subprocess.run(cmd, cwd=str(work_dir), text=True, capture_output=True)

    row = {
        "dataset": Path(dataset).parts[-2] if len(Path(dataset).parts) >= 2 else dataset,
        "question_file": dataset,
        "status": "ok" if proc.returncode == 0 else "failed",
        "throughput": "",
        "accept_length": "",
        "error": "",
    }
    if proc.returncode == 0:
        try:
            result = parse_result_jsonl(tmp_path)
            row["throughput"] = result.get("throughput", "")
            row["accept_length"] = result.get("accept_length", "")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"parse result failed: {exc}"
    else:
        row["error"] = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:500]

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return row


def build_server_cmd(
    args: argparse.Namespace,
    algorithm: str,
    draft_model_path: str,
    max_running_requests: int,
) -> list[str]:
    # Keep user-facing naming as PRISM, but sglang still expects LD.
    sglang_algorithm = "LD" if algorithm == "PRISM" else algorithm
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--dtype",
        "bfloat16",
        "--cuda-graph-max-bs",
        "1",
        "--max-running-requests",
        str(max_running_requests),
        "--port",
        str(args.port),
    ]
    if algorithm != "NONE":
        cmd.extend(
            [
                "--speculative-algorithm",
                sglang_algorithm,
                "--speculative-draft-model-path",
                draft_model_path,
                "--speculative-num-steps",
                "6",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "16",
            ]
        )
    return cmd


def safe_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "evaluation" / "client" / "bench_mt_all.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate repo root from {start}; expected evaluation/client/bench_mt_all.py"
    )


def resolve_under_script_dir(script_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return script_dir / p


def main() -> int:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    repo_root = find_repo_root(script_dir)
    default_model_prefix = str((repo_root / "models").resolve())
    parser = argparse.ArgumentParser(
        description="Sweep max-running-requests and compute average throughput across datasets."
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--output-csv", default="e2e_llama2_bs_sweep_avg.csv")
    parser.add_argument("--detail-csv", default="e2e_llama2_bs_sweep_detail.csv")
    parser.add_argument("--server-log-dir", default="logs")
    parser.add_argument("--model-path", default=str(Path(default_model_prefix) / "Llama-2-7b-chat-hf"))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["NONE", "EAGLE", "PRISM"],
        help="Speculative algorithms to test. Example: NONE EAGLE PRISM",
    )
    parser.add_argument(
        "--draft-model-paths",
        nargs="+",
        default=["-", "EAGLE2-800K-SGLANG/llama2-7b", "LD-800k"],
        help="One-to-one with --algorithms; for NONE use '-'.",
    )
    parser.add_argument(
        "--draft-model-prefix",
        default=default_model_prefix,
        help="Common prefix for draft model paths. Applied to non-absolute entries except '-'.",
    )
    parser.add_argument("--batch-size", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    client_dir = repo_root / "evaluation" / "client"
    bench_script = client_dir / "bench_mt_all.py"
    if not bench_script.exists():
        print(f"[Error] bench script not found: {bench_script}", file=sys.stderr)
        return 1

    out_avg_csv = resolve_under_script_dir(script_dir, args.output_csv)
    out_detail_csv = resolve_under_script_dir(script_dir, args.detail_csv)
    log_dir = resolve_under_script_dir(script_dir, args.server_log_dir)
    out_avg_csv.parent.mkdir(parents=True, exist_ok=True)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    algorithms = [normalize_algorithm(a) for a in args.algorithms]
    if len(algorithms) != len(args.draft_model_paths):
        print("[Error] --draft-model-paths must have the same length as --algorithms", file=sys.stderr)
        return 1
    draft_paths = [
        with_draft_model_prefix(normalized, args.draft_model_prefix) if normalized else ""
        for normalized in (normalize_draft_path(p) for p in args.draft_model_paths)
    ]
    for algo, draft_path in zip(algorithms, draft_paths):
        if algo != "NONE" and not draft_path:
            print(f"[Error] algorithm={algo} requires a valid draft model path", file=sys.stderr)
            return 1

    all_detail_rows = []
    all_summary_rows = []

    try:
        for max_rr in args.max_running_requests:
            for algorithm, draft_path in zip(algorithms, draft_paths):
                server_proc = None
                run_key = f"{algorithm.lower()}_mrr{max_rr}"
                server_log_path = log_dir / f"e2e_llama2_bs_sweep_{run_key}.log"
                server_cmd = build_server_cmd(args, algorithm, draft_path, max_rr)

                print(f"\n=== Algorithm={algorithm}, max-running-requests={max_rr} ===")
                print(f"Draft model: {draft_path or '(none)'}")
                print(f"Server log: {server_log_path}")

                per_run_rows = []
                startup_failed = False
                with open(server_log_path, "w", encoding="utf-8") as server_log:
                    print("[1/3] Starting sglang server...")
                    server_proc = subprocess.Popen(
                        server_cmd,
                        cwd=str(repo_root),
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                    )

                    print(f"[2/3] Waiting for server on {args.port}...")
                    if not wait_for_port("127.0.0.1", args.port, args.startup_timeout):
                        print("[Error] Server startup timeout. Check server log.", file=sys.stderr)
                        startup_failed = True
                        for dataset in args.datasets:
                            per_run_rows.append(
                                {
                                    "algorithm": algorithm,
                                    "max_running_requests": max_rr,
                                    "dataset": Path(dataset).parts[-2]
                                    if len(Path(dataset).parts) >= 2
                                    else dataset,
                                    "question_file": dataset,
                                    "status": "failed",
                                    "throughput": "",
                                    "accept_length": "",
                                    "error": "server startup timeout",
                                }
                            )
                    else:
                        print("[3/3] Running datasets...")
                        for dataset in args.datasets:
                            print(f"- {dataset}")
                            one = run_one_dataset(
                                python_exe=sys.executable,
                                bench_script=bench_script,
                                work_dir=client_dir,
                                dataset=dataset,
                                port=args.port,
                                temperature=args.temperature,
                                num_questions=args.num_questions,
                            )
                            one["algorithm"] = algorithm
                            one["max_running_requests"] = max_rr
                            per_run_rows.append(one)

                if server_proc is not None:
                    print("Stopping server...")
                    terminate_process(server_proc)
                    time.sleep(1)

                throughputs = [safe_float(r.get("throughput")) for r in per_run_rows if r.get("status") == "ok"]
                throughputs = [x for x in throughputs if x is not None]
                avg_throughput = sum(throughputs) / len(throughputs) if throughputs else ""
                ok_count = sum(1 for r in per_run_rows if r.get("status") == "ok")

                summary_row = {
                    "algorithm": algorithm,
                    "max_running_requests": max_rr,
                    "num_datasets": len(per_run_rows),
                    "num_success": ok_count,
                    "avg_throughput": avg_throughput,
                    "status": "ok"
                    if (not startup_failed and ok_count == len(per_run_rows) and len(per_run_rows) > 0)
                    else "failed",
                    "error": "" if not startup_failed else "server startup timeout",
                }
                all_detail_rows.extend(per_run_rows)
                all_summary_rows.append(summary_row)
                print(f"Average throughput: {avg_throughput} (success {ok_count}/{len(per_run_rows)})")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    with open(out_detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "max_running_requests",
                "dataset",
                "question_file",
                "status",
                "throughput",
                "accept_length",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(all_detail_rows)

    with open(out_avg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "max_running_requests",
                "num_datasets",
                "num_success",
                "avg_throughput",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(all_summary_rows)

    ok_runs = sum(1 for r in all_summary_rows if r["status"] == "ok")
    print(f"Done. Successful runs: {ok_runs}/{len(all_summary_rows)}.")
    print(f"Average CSV: {out_avg_csv}")
    print(f"Detail CSV: {out_detail_csv}")
    return 0 if ok_runs == len(all_summary_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
