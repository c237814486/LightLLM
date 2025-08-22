export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev:$PYTHONPATH"


python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir /mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/checkpoints/omni_models/0812_llava_omni_qwen25vl_14B_16x_4k_st2_kimiwhisper_10x_unfreezeaudio_omnidata_text500w_lr2e-6_audiotoken_8k \
    --max_req_total_len 8192 \
    --max_total_token_num 32678 \
    --cache_capacity 20000 \
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
    --quant_type  vllm-fp8w8a8 \
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 8 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --enable_multimodal \
    --enable_multimodal_audio \
    --graph_max_batch_size 4 \
    --graph_max_len_in_batch 8192 \
    --visual_gpu_ids 1 \
    --audio_gpu_ids 1

