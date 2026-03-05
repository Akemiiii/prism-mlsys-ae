#!/usr/bin/env python3
"""查看服务器信息并提取 CUDA Graph batch sizes"""
import requests
import json

url = "http://127.0.0.1:30003/get_server_info"
response = requests.get(url)
data = response.json()

print("=" * 70)
print("检查 internal_states:")
print("=" * 70)

if "internal_states" in data:
    print(f"找到 internal_states，共 {len(data['internal_states'])} 个")
    for idx, state in enumerate(data['internal_states']):
        print(f"\n--- State {idx} ---")
        if "cuda_graph_batch_sizes" in state:
            print(f"✓ 找到 cuda_graph_batch_sizes:")
            print(f"  {state['cuda_graph_batch_sizes']}")
        else:
            print("✗ 未找到 cuda_graph_batch_sizes")
            print(f"  可用的键: {list(state.keys())[:10]}...")  # 显示前10个键
else:
    print("✗ 未找到 internal_states")
    print(f"顶层键: {list(data.keys())}")


