#!/usr/bin/env python3
"""依次测试六个数据集"""
import subprocess
import argparse

# 六个数据集
datasets = [
    "data/qa/question.jsonl",
    #"data/mt_bench/question.jsonl",
   # "data/humaneval/question.jsonl",
  #  "data/gsm8k/question.jsonl",
 #   "data/alpaca/question.jsonl",
    "data/sum/question.jsonl",
    "data/qa/question.jsonl"
]

def main():
    parser = argparse.ArgumentParser(description="依次测试多个数据集")
    parser.add_argument("--temperature", type=float, default=0, help="温度参数")
    parser.add_argument("--port", type=int, default=30000, help="服务器端口")
    parser.add_argument("--num-questions", type=int, default=80, help="每个数据集测试的问题数量")
    args = parser.parse_args()
    
    result_file = f"results_temp_{args.temperature}.jsonl"
    
    # 依次运行
    for dataset in datasets:
        print(f"\n测试: {dataset}")
        cmd = [
            "python3", "bench_mt.py",
            "--question-file", dataset,
            "--temperature", str(args.temperature),
            "--result-file", result_file,
            "--port", str(args.port),
            "--num-questions", str(args.num_questions)
        ]
        subprocess.run(cmd, check=True)
    
    print(f"\n完成！结果保存在: {result_file}")

if __name__ == "__main__":
    main()
