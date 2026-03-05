#!/bin/bash
# 诊断脚本：检查bench测试状态

echo "=== 检查 bench_mt.py 进程 ==="
ps aux | grep "bench_mt.py" | grep -v grep

echo ""
echo "=== 检查 sglang 服务器进程 ==="
ps aux | grep "sglang.launch_server" | grep -v grep

echo ""
echo "=== 检查端口占用情况 ==="
netstat -tlnp 2>/dev/null | grep -E "30000|30001|30002" || ss -tlnp | grep -E "30000|30001|30002"

echo ""
echo "=== 测试服务器是否响应 (端口 30000) ==="
curl -s http://127.0.0.1:30000/health 2>&1 | head -n 5

echo ""
echo "=== 最近的结果文件更新时间 ==="
ls -lh EAGLE\(HASS\)_llama2_temp_0_result.jsonl 2>/dev/null || echo "结果文件不存在"

