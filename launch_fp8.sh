export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev:$PYTHONPATH"
set -x
# ./dist/llm_tts_server/llm_tts_server \
python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir /mnt/afs/lijiayi1/code/game_video/test/20250813_beebee \
    --max_req_total_len 4000 \
    --max_total_token_num 4096 \
    --cache_capacity 12000 \
    --mode ppl_int8kv_flashdecoding \
    --data_type bf16 \
    --port 18003 \
    --tokenizer_mode auto \
    --trust_remote_code \
    --host 0.0.0.0 \
    --use_dynamic_prompt_cache \
    --tp 1 \
    --nccl_port 28765 \
    --mem_fraction 0.9 \
    --quant_type vllm-fp8w8a8 \
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 8 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --enable_multimodal \
    --graph_max_batch_size 4 \
    --graph_max_len_in_batch 1024 \
    --visual_gpu_ids 0 \
    --audio_gpu_ids 0 \
    --chunked_prefill_size 1024 \
    # --service llm \

