#!/usr/bin/env python3
"""
在单个GPU上运行所有基准测试的脚本
启动一个服务器，然后顺序测试所有数据集
"""
import os
import subprocess
import time
import json
import sys
import requests


def check_server_health(port, timeout=300, interval=5):
    """
    检查服务器是否健康
    
    Args:
        port: 服务器端口
        timeout: 最大等待时间（秒）
        interval: 检查间隔（秒）
    
    Returns:
        True 如果服务器健康，False 否则
    """
    print(f"检查服务器健康状态（端口 {port}）...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ 服务器已就绪！")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        print(f"  等待中... ({int(time.time() - start_time)}s)", end='\r')
        time.sleep(interval)
    print(f"\n✗ 服务器启动超时")
    return False


def start_server(gpu_id, port, model_path, draft_model_path, context_length=2048):
    """在指定GPU上启动服务器"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    server_cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--dtype", "bfloat16",
        "--context-length", str(context_length),
        "--speculative-algorithm", "EAGLE",
        "--speculative-draft-model-path", draft_model_path,
        "--speculative-num-steps", "6",
        "--speculative-eagle-topk", "4",
        "--speculative-num-draft-tokens", "16",
        "--cuda-graph-max-bs", "1",
        "--max-running-requests", "1",
        "--port", str(port)
    ]
    
    print(f"\n{'='*60}")
    print(f"在 GPU {gpu_id} 上启动服务器（端口 {port}）")
    print(f"{'='*60}")
    print(f"命令: {' '.join(server_cmd)}\n")
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    return server_process


def run_benchmark(question_file, temperature, result_file, port):
    """运行单个基准测试"""
    dataset_name = question_file.split('/')[-2]  # 从路径中提取数据集名称
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset_name} ({question_file})")
    print(f"{'='*60}")
    
    cmd = [
        "python3",
        "bench_mt_all.py",
        "--question-file", question_file,
        "--temperature", str(temperature),
        "--result-file", result_file,
        "--port", str(port),
        "--num-questions", "80"
    ]
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ 完成 {dataset_name} 测试 (耗时: {elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {dataset_name} 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ {dataset_name} 意外错误: {e}")
        return False


def main():
    # ==================== 配置参数 ====================
    gpu_id = 0  # 使用的 GPU ID
    port = 30000  # 服务器端口
    temperature = 0  # 温度参数
    context_length = 2048  # 上下文长度
    
    # 模型路径
    model_path = "/mnt/data1/ytchen/cache/Meta-Llama-3-8B-Instruct"
    draft_model_path = "/mnt/data1/ytchen/cache/FangLIU/LD2/llama3-8b/Eagle2-800k"
    
    # 结果文件
    result_file = f"EAGLE_llama3_temp_{temperature}_all_datasets.jsonl"
    
    # 所有测试数据集
    question_files = [
        # "data/mt_bench/question.jsonl",
        # "data/humaneval/question.jsonl",
        # "data/gsm8k/question.jsonl",
        # "data/alpaca/question.jsonl",
        "data/sum/question.jsonl",
        "data/qa/question.jsonl"
    ]
    
    print(f"\n{'#'*60}")
    print(f"# 单GPU基准测试")
    print(f"#")
    print(f"# GPU ID: {gpu_id}")
    print(f"# 端口: {port}")
    print(f"# 温度: {temperature}")
    print(f"# 上下文长度: {context_length}")
    print(f"# 数据集数量: {len(question_files)}")
    print(f"# 结果文件: {result_file}")
    print(f"{'#'*60}\n")
    
    # 初始化结果文件，写入配置信息
    with open(result_file, "w") as f:
        config_info = {
            "model_path": model_path,
            "draft_model_path": draft_model_path,
            "gpu_id": gpu_id,
            "port": port,
            "temperature": temperature,
            "context_length": context_length,
            "datasets": question_files,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        f.write(json.dumps(config_info, indent=2) + "\n")
        f.write("-" * 60 + "\n")
    
    print(f"配置信息已写入: {result_file}\n")
    
    # 启动服务器
    server_process = start_server(gpu_id, port, model_path, draft_model_path, context_length)
    
    try:
        # 等待服务器启动并检查健康状态
        if not check_server_health(port, timeout=300, interval=5):
            print("\n服务器启动失败，退出程序")
            # 打印服务器错误输出
            stderr = server_process.stderr.read()
            if stderr:
                print(f"\n服务器错误输出:\n{stderr[:2000]}")
            server_process.terminate()
            sys.exit(1)
        
        # 顺序运行所有基准测试
        print(f"\n开始运行基准测试...")
        successful = 0
        failed = 0
        
        for idx, question_file in enumerate(question_files, 1):
            print(f"\n进度: [{idx}/{len(question_files)}]")
            
            if run_benchmark(question_file, temperature, result_file, port):
                successful += 1
            else:
                failed += 1
        
        # 打印总结
        print(f"\n{'='*60}")
        print(f"测试完成！")
        print(f"{'='*60}")
        print(f"成功: {successful}/{len(question_files)}")
        print(f"失败: {failed}/{len(question_files)}")
        print(f"结果保存在: {result_file}")
        print(f"{'='*60}\n")
    
    finally:
        # 关闭服务器
        print("\n关闭服务器...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
            print("服务器已关闭")
        except subprocess.TimeoutExpired:
            print("服务器未能正常关闭，强制终止...")
            server_process.kill()
            print("服务器已强制关闭")


if __name__ == "__main__":
    main()

