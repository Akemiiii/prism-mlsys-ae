#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path /mnt/data1/ytchen/cache/Llama-2-7b-chat-hf \
    --dtype bfloat16 \
    --speculative-algorithm LD \
    --speculative-draft-model-path /mnt/data1/ytchen/cache/LD/LD2 \
    --speculative-num-steps 6 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --cuda-graph-max-bs 32 \
    --max-running-requests 32 \
    --port 30003


