#!/usr/bin/env python3
"""打印 cuda-graph-bs 的默认列表"""

def get_cuda_graph_bs_list(speculative_algorithm=None, disable_cuda_graph_padding=False, gpu_mem_gb=None):
    """
    计算 cuda-graph-bs 的默认列表
    
    Args:
        speculative_algorithm: 推测解码算法 (None, "LD", "EAGLE", 等)
        disable_cuda_graph_padding: 是否禁用 CUDA Graph padding
        gpu_mem_gb: GPU 显存大小（GB）
    """
    if speculative_algorithm is None:
        if disable_cuda_graph_padding:
            capture_bs = list(range(1, 33)) + list(range(40, 161, 16))
        else:
            capture_bs = [1, 2, 4, 8] + list(range(16, 161, 8))
    else:
        # 推测解码需要更多 CUDA Graph 内存，所以捕获更少
        capture_bs = (
            list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 161, 16))
        )
    
    # 大显存 GPU
    if gpu_mem_gb is not None and gpu_mem_gb > 80:
        capture_bs += list(range(160, 257, 8))
    
    return sorted(set(capture_bs))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="打印 cuda-graph-bs 默认列表")
    parser.add_argument("--speculative-algorithm", type=str, default=None,
                       help="推测解码算法 (LD, EAGLE, EAGLE3, 等)")
    parser.add_argument("--disable-cuda-graph-padding", action="store_true",
                       help="禁用 CUDA Graph padding")
    parser.add_argument("--gpu-mem-gb", type=int, default=None,
                       help="GPU 显存大小（GB）")
    args = parser.parse_args()
    
    bs_list = get_cuda_graph_bs_list(
        args.speculative_algorithm,
        args.disable_cuda_graph_padding,
        args.gpu_mem_gb
    )
    
    print(f"配置:")
    print(f"  speculative_algorithm: {args.speculative_algorithm}")
    print(f"  disable_cuda_graph_padding: {args.disable_cuda_graph_padding}")
    print(f"  gpu_mem_gb: {args.gpu_mem_gb}")
    print(f"\ncuda-graph-bs 列表 (共 {len(bs_list)} 个):")
    print(bs_list)
    print(f"\n命令行格式:")
    print(f"--cuda-graph-bs {' '.join(map(str, bs_list))}")


