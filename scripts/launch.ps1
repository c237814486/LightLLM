# ==========================================================
# 1. 设置 PYTHONPATH 环境变量 
# ==========================================================
$projectRoot = "D:\LeapFaith\LightLLM"
$env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:USE_LIBUV = "0"

Write-Host "Setting PYTHONPATH: $env:PYTHONPATH"

# # ==========================================================
# # 2. 运行 Python 命令
# # PowerShell 可以使用反引号 ` 进行命令换行
# # ==========================================================
# python -m lightllm.server.api_server `
#     --zmq_mode "tcp://" `
    # --mode triton_gqa_flashdecoding `
#     --sampling_backend triton_top_pk `
#     --model_dir "D:\LeapFaith\models\Qwen2.5-VL-3B-Instruct" `
#     --enable_multimodal `
#     --host 0.0.0.0 `
#     --port 10083 `
#     --running_max_req_size 1 `
#     --chunked_prefill_size 2048 `
#     --max_req_total_len 4080 `
#     --max_total_token_num 4096 `
#     --quant_type "ao-fp8w8a16" `
#     --vit_quant_type "ao-fp8w8a16" `
#     --mem_fraction 0.95

# ==========================================================
# 2. 运行 Python 命令
# PowerShell 可以使用反引号 ` 进行命令换行
# ==========================================================
python -m lightllm.server.api_server `
    --zmq_mode "tcp://" `
    --mode triton_gqa_flashdecoding `
    --sampling_backend triton_top_pk `
    --model_dir "D:\LeapFaith\models\0911_llava_omni_qwen25vl_7B_st2_4k_800w_audio_omni" `
    --host 0.0.0.0 `
    --port 10083 `
    --running_max_req_size 1 `
    --chunked_prefill_size 2048 `
    --max_req_total_len 4080 `
    --max_total_token_num 4096 `
    --mem_fraction 0.95 `
    --quant_type "ao-fp8w8a16" `
    --vit_quant_type "ao-fp8w8a16" `
    --enable_multimodal