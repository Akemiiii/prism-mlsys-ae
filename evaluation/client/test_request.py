#!/usr/bin/env python3
"""
手动测试 SGLang 服务器的脚本
"""
import requests
import json

def test_server(port=30001):
    """测试服务器是否正常工作"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": "Meta-Llama-3-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "你好，请介绍一下自己。"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    print(f"发送请求到端口 {port}...")
    print(f"请求内容: {json.dumps(payload, indent=2, ensure_ascii=False)}\n")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 请求成功！\n")
            print("=" * 60)
            print("模型回复:")
            print("=" * 60)
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(content)
                print("\n" + "=" * 60)
                
                # 打印统计信息
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\nToken 使用统计:")
                    print(f"  输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"  输出 tokens: {usage.get('completion_tokens', 'N/A')}")
                    print(f"  总计 tokens: {usage.get('total_tokens', 'N/A')}")
            else:
                print("响应格式异常:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"✗ 请求失败！状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"✗ 连接失败！请确保服务器在端口 {port} 上运行")
    except requests.exceptions.Timeout:
        print("✗ 请求超时！")
    except Exception as e:
        print(f"✗ 发生错误: {e}")


def test_health(port=30001):
    """检查服务器健康状态"""
    url = f"http://localhost:{port}/health"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ 服务器健康检查通过 (端口 {port})")
            return True
        else:
            print(f"✗ 服务器健康检查失败 (端口 {port})")
            return False
    except:
        print(f"✗ 无法连接到服务器 (端口 {port})")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 SGLang 服务器")
    parser.add_argument("--port", type=int, default=30001, help="服务器端口")
    parser.add_argument("--health-only", action="store_true", help="只检查健康状态")
    args = parser.parse_args()
    
    if args.health_only:
        test_health(args.port)
    else:
        # 先检查健康状态
        if test_health(args.port):
            print()
            test_server(args.port)
        else:
            print("\n服务器未就绪，跳过测试")

