export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev:$PYTHONPATH"

# /usr/bin/env /mnt/afs/yangdeyu/conda_env/llava/bin/python -m lightllm.server.api_server \
#     --run_mode normal \
#     --model_dir /mnt/afs/share/Qwen25-7B-Instruct \
#     --max_req_total_len 8192 \
#     --max_total_token_num 120000 \
#     --cache_capacity 12000 \
#     --mode triton_gqa_flashdecoding \
#     --data_type bf16 \
#     --port 18003 \
#     --tokenizer_mode auto \
#     --trust_remote_code \
#     --host 0.0.0.0 \
#     --use_dynamic_prompt_cache \
#     --tp 1 \
#     --nccl_port 28765 \
#     --mem_fraction 0.9 \
#     --graph_max_batch_size 32 \
#     --graph_max_len_in_batch 8192 \
#     --sampling_backend sglang_kernel \
    # --quant_type vllm-w8a8-perchannel


/usr/bin/env /mnt/afs/yangdeyu/conda_env/llava/bin/python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir /mnt/afs/lijiayi1/code/game_video/test/20250813_beebee \
    --max_req_total_len 8192 \
    --max_total_token_num 120000 \
    --cache_capacity 12000 \
    --mode triton_gqa_flashdecoding \
    --data_type bf16 \
    --port 18003 \
    --tokenizer_mode auto \
    --trust_remote_code \
    --host 0.0.0.0 \
    --use_dynamic_prompt_cache \
    --tp 1 \
    --nccl_port 28765 \
    --mem_fraction 0.9 \
    --graph_max_batch_size 32 \
    --graph_max_len_in_batch 8192 \
    --enable_multimodal \
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 16 \
    --sampling_backend triton \
    # --enable_concurrent_alloc \
