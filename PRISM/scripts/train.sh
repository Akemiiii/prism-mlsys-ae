CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision=bf16 train/main_LD.py \
    --basepath Model_weights/Llama-2-7b-chat-hf/ \
    --tmpdir train_data/ShareGPT/sharegpt_0_67999_mufp16/ \
    --cpdir reproduce_hass \
    --configpath train/LD_llama_2_7B_config.json \
    --epoch 40 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3
