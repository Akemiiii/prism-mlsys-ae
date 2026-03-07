#!/usr/bin/env python3
"""Start sglang, evaluate multiple datasets, and save results to CSV."""

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
        "algorithm": "",
        "draft_model_path": "",
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


def build_server_cmd(args: argparse.Namespace, algorithm: str, draft_model_path: str) -> list[str]:
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
        "1",
        "--port",
        str(args.port),
    ]
    if algorithm != "NONE":
        cmd.extend(
            [
                "--speculative-algorithm",
                algorithm,
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


def make_algo_log_path(base_log: Path, algorithm: str, multiple: bool) -> Path:
    if not multiple:
        return base_log
    suffix = base_log.suffix or ".log"
    return base_log.with_name(f"{base_log.stem}_{algorithm.lower()}{suffix}")


def algorithm_name_for_csv(draft_model_path: str, fallback_algorithm: str) -> str:
    p = draft_model_path.upper()
    if "HASS" in p:
        return "HASS"
    if "EAGLE" in p:
        return "EAGLE"
    if "LD" in p or "PRISM" in p:
        return "PRISM"
    if fallback_algorithm.upper() == "NONE":
        return "NONE"
    return fallback_algorithm


def main() -> int:
    default_model_prefix = str((Path(__file__).resolve().parent / "../models").resolve())
    parser = argparse.ArgumentParser(description="Start sglang, run multi-dataset evaluation, and write CSV.")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--output-csv", default="evaluation/e2e_llama2.csv")
    parser.add_argument("--server-log", default="evaluation/logs/e2e_llama2_server.log")
    parser.add_argument("--model-path", default=str(Path(default_model_prefix) / "Llama-2-7b-chat-hf"))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["NONE", "EAGLE", "EAGLE", "LD"],
        help="Examples: NONE LD EAGLE",
    )
    parser.add_argument(
        "--draft-model-paths",
        nargs="+",
        default=["-", "EAGLE2-800K-SGLANG/llama2-7b", "HASS-800K-SGLANG/llama2-7b", "LD-800k"],
        help="One-to-one with --algorithms; for NONE use '-'. Non-absolute paths use --draft-model-prefix.",
    )
    parser.add_argument(
        "--draft-model-prefix",
        default=default_model_prefix,
        help="Common prefix for draft model paths. Applied to each --draft-model-paths entry except '-' and absolute paths.",
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    client_dir = repo_root / "evaluation" / "client"
    bench_script = client_dir / "bench_mt_all.py"
    if not bench_script.exists():
        print(f"[Error] bench script not found: {bench_script}", file=sys.stderr)
        return 1

    log_path = repo_root / args.server_log
    csv_path = repo_root / args.output_csv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    algorithms = [normalize_algorithm(a) for a in args.algorithms]
    if len(args.draft_model_paths) != len(algorithms):
        print("[Error] --draft-model-paths must have the same length as --algorithms", file=sys.stderr)
        return 1
    draft_paths = [
        with_draft_model_prefix(normalized, args.draft_model_prefix) if normalized else ""
        for normalized in (normalize_draft_path(p) for p in args.draft_model_paths)
    ]

    for algo, draft_path in zip(algorithms, draft_paths):
        if not algo:
            print("[Error] algorithm cannot be empty", file=sys.stderr)
            return 1
        if algo != "NONE" and not draft_path:
            print(f"[Error] algorithm={algo} requires a valid draft model path", file=sys.stderr)
            return 1
    
    if not algorithms:
        print("[Error] At least one algorithm is required", file=sys.stderr)
        return 1

    try:
        for algorithm, draft_path in zip(algorithms, draft_paths):
            server_proc = None
            csv_algorithm = algorithm_name_for_csv(draft_path, algorithm)
            algo_log_path = make_algo_log_path(log_path, algorithm, multiple=len(algorithms) > 1)
            server_cmd = build_server_cmd(args, algorithm, draft_path)
            print(f"\n=== Algorithm: {algorithm} ===")
            print(f"Draft model: {draft_path or '(none)'}")
            print(f"Server log: {algo_log_path}")

            with open(algo_log_path, "w", encoding="utf-8") as server_log:
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
                    for dataset in args.datasets:
                        rows.append(
                            {
                                "algorithm": csv_algorithm,
                                "draft_model_path": draft_path,
                                "dataset": Path(dataset).parts[-2] if len(Path(dataset).parts) >= 2 else dataset,
                                "question_file": dataset,
                                "status": "failed",
                                "throughput": "",
                                "accept_length": "",
                                "error": "server startup timeout",
                            }
                        )
                    continue

                print("[3/3] Running datasets...")
                for dataset in args.datasets:
                    print(f"- {dataset}")
                    row = run_one_dataset(
                        python_exe=sys.executable,
                        bench_script=bench_script,
                        work_dir=client_dir,
                        dataset=dataset,
                        port=args.port,
                        temperature=args.temperature,
                        num_questions=args.num_questions,
                    )
                    row["algorithm"] = csv_algorithm
                    row["draft_model_path"] = draft_path
                    rows.append(row)

            if server_proc is not None:
                print("Stopping server...")
                terminate_process(server_proc)
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "draft_model_path",
                "dataset",
                "question_file",
                "status",
                "throughput",
                "accept_length",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Done. {ok}/{len(rows)} datasets succeeded. CSV: {csv_path}")
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
