export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev:$PYTHONPATH"
set -x
# ./dist/llm_tts_server/llm_tts_server \
# source activate /mnt/afs/yangdeyu/conda_env/llava_4090
export PATH="/root/miniconda3/envs/llava/bin:$PATH"
which gunicorn

/usr/bin/env /root/miniconda3/envs/llava/bin/python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir /mnt/afs/lijiayi1/code/game_video/test/20250921_beebee_audio_4node \
    --max_req_total_len 8192 \
    --max_total_token_num 16000 \
    --cache_capacity 15000 \
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
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 8 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --enable_multimodal \
    --enable_multimodal_audio \
    --graph_max_batch_size 2 \
    --graph_max_len_in_batch 4096 \
    --visual_gpu_ids 1 \
    --audio_gpu_ids 1 \
    --chunked_prefill_size 4096 \
    --quant_type vllm-fp8w8a8 \
    # --service llm \

