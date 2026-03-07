#!/usr/bin/env python3
"""Start sglang, evaluate multiple datasets, and insert results into existing CSV."""

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

CSV_FIELDS = [
    "algorithm",
    "draft_model_path",
    "dataset",
    "question_file",
    "status",
    "throughput",
    "accept_length",
    "error",
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
        "algorithm": "Standalone",
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


def with_draft_model_prefix(path: str, prefix: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(prefix) / p)


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    algo = args.speculative_algorithm.strip().upper()
    return [
        "env",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1",
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
        "--attention-backend",
        "fa3",
        "--port",
        str(args.port),
        "--speculative-algorithm",
        algo,
        "--speculative-draft-model-path",
        args.draft_model_path,
        "--speculative-num-steps",
        "6",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "16",
    ]


def insert_rows_after_header(csv_path: Path, new_rows: list[dict]) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = reader.fieldnames
        existing_rows = list(reader)

    # Ensure all required fields for this evaluation are present.
    missing = [field for field in CSV_FIELDS if field not in fieldnames]
    if missing:
        raise ValueError(f"CSV header missing fields: {missing}")

    none_indices = [idx for idx, row in enumerate(existing_rows) if str(row.get("algorithm", "")).upper() == "NONE"]
    eagle_indices = [idx for idx, row in enumerate(existing_rows) if str(row.get("algorithm", "")).upper() == "EAGLE"]
    if not none_indices:
        raise ValueError("CSV does not contain algorithm=NONE")
    if not eagle_indices:
        raise ValueError("CSV does not contain algorithm=EAGLE")
    insert_at = eagle_indices[0]
    if max(none_indices) >= insert_at:
        raise ValueError("Cannot insert between NONE and EAGLE because row order is unexpected")

    merged_rows = existing_rows[:insert_at] + new_rows + existing_rows[insert_at:]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            normalized = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(normalized)


def main() -> int:
    default_model_prefix = str((Path(__file__).resolve().parent / "../models").resolve())
    parser = argparse.ArgumentParser(
        description="Start sglang, run multi-dataset evaluation, and insert rows into an existing CSV."
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--output-csv", default="evaluation/e2e_llama3.csv")
    parser.add_argument("--server-log", default="evaluation/logs/e2e_llama3_standalone_server.log")
    parser.add_argument("--model-path", default=str(Path(default_model_prefix) / "Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--speculative-algorithm", default="STANDALONE")
    parser.add_argument(
        "--draft-model-path",
        default=with_draft_model_prefix("Llama-3.2-1B-Instruct", default_model_prefix),
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

    rows = []
    server_cmd = build_server_cmd(args)
    server_proc = None

    try:
        print(f"Speculative algorithm: {args.speculative_algorithm}")
        print(f"Draft model: {args.draft_model_path}")
        print(f"Server log: {log_path}")

        with open(log_path, "w", encoding="utf-8") as server_log:
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
                            "algorithm": args.speculative_algorithm,
                            "draft_model_path": args.draft_model_path,
                            "dataset": Path(dataset).parts[-2] if len(Path(dataset).parts) >= 2 else dataset,
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
                    row = run_one_dataset(
                        python_exe=sys.executable,
                        bench_script=bench_script,
                        work_dir=client_dir,
                        dataset=dataset,
                        port=args.port,
                        temperature=args.temperature,
                        num_questions=args.num_questions,
                    )
                    row["algorithm"] = args.speculative_algorithm
                    row["draft_model_path"] = args.draft_model_path
                    rows.append(row)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        if server_proc is not None:
            print("Stopping server...")
            terminate_process(server_proc)
            time.sleep(1)

    try:
        insert_rows_after_header(csv_path, rows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Done. {ok}/{len(rows)} datasets succeeded. Inserted rows into CSV: {csv_path}")
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
