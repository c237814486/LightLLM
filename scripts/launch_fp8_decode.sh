export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev:$PYTHONPATH"

# decode 节点不需要multimodal
python -m lightllm.server.api_server \
    --run_mode decode \
    --model_dir /mnt/afs/lijiayi1/code/game_video/test/20250813_beebee \
    --pd_master_ip 0.0.0.0 \
    --pd_master_port 18003 \
    --pd_decode_rpyc_port 42000 \
    --max_req_total_len 8192 \
    --max_total_token_num 16384 \
    --cache_capacity 20000 \
    --mode triton_flashdecoding \
    --data_type bf16 \
    --port 18005 \
    --tokenizer_mode auto \
    --trust_remote_code \
    --host 0.0.0.0 \
    --use_dynamic_prompt_cache \
    --tp 1 \
    --nccl_port 28768 \
    --mem_fraction 0.9 \
    --quant_type  vllm-fp8w8a8 \
    --visual_nccl_ports 29502 \
    --visual_infer_batch_size 8 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --graph_max_batch_size 4 \
    --graph_max_len_in_batch 8192 \

