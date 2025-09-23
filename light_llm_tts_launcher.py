#!/usr/bin/env python3
"""
统一启动器 - 支持启动TTS或LLM服务
根据命令行参数选择启动不同的服务
"""

import os
import sys
import argparse
import logging
import warnings
import subprocess
from pathlib import Path
from typing import Optional

# 抑制警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TYPEGUARD_DISABLE", "true")

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("launcher")


def get_exe_base_dir() -> str:
    """返回冻结/开发环境下的基准目录"""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


def prepare_common_env(env: dict) -> dict:
    """准备通用运行环境变量"""
    # base_dir = get_exe_base_dir()

    # 抑制警告
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"
    env["TYPEGUARD_DISABLE"] = "true"

    # 设置CUDA相关
    env.setdefault("TORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # 设置多进程相关
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    return env


def prepare_tts_env(env: dict) -> dict:
    """准备TTS特定环境变量"""
    base_dir = get_exe_base_dir()

    # 设置light_tts路径
    light_tts_path = os.path.join(base_dir, "light_tts")

    if os.path.exists(light_tts_path):
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{light_tts_path}:{current_pythonpath}"
            if current_pythonpath
            else light_tts_path
        )
        logger.info(f"设置light_tts路径: {light_tts_path}")

    # 设置cosyvoice路径
    cosyvoice_path = os.path.join(base_dir, "cosyvoice")
    if os.path.exists(cosyvoice_path):
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{cosyvoice_path}:{current_pythonpath}"
            if current_pythonpath
            else cosyvoice_path
        )
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"
    # 设置pretrained_models路径
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            pretrained_models_path = os.path.join(base_dir, "pretrained_models")
        else:
            exe_parent_dir = os.path.dirname(base_dir)
            pretrained_models_path = os.path.join(exe_parent_dir, "pretrained_models")

        if os.path.exists(pretrained_models_path):
            env["COSYVOICE_PRETRAINED_MODELS"] = os.path.abspath(pretrained_models_path)
            logger.info(
                f"设置pretrained_models路径: {os.path.abspath(pretrained_models_path)}"
            )

        # 设置assets路径
        assets_path = os.path.join(base_dir, "assets")
        if os.path.exists(assets_path):
            env["COSYVOICE_PROMPT_AUDIO"] = os.path.join(assets_path, "prompt_audio")

    return env


def prepare_llm_env(env: dict) -> dict:
    """准备LLM特定环境变量"""
    base_dir = get_exe_base_dir()

    # 设置lightllm路径
    lightllm_path = os.path.join(base_dir, "lightllm")
    if os.path.exists(lightllm_path):
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{lightllm_path}:{current_pythonpath}"
            if current_pythonpath
            else lightllm_path
        )
        logger.info(f"设置lightllm路径: {lightllm_path}")

    # 配置lightllm_kernel路径
    if getattr(sys, "frozen", False):
        kernel_pkg_dir = os.path.join(base_dir, "_internal", "lightllm_kernel")
    else:
        kernel_pkg_dir = os.path.join(base_dir, "lightllm_kernel")
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"
    if os.path.exists(kernel_pkg_dir):
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{kernel_pkg_dir}:{current_pythonpath}"
            if current_pythonpath
            else kernel_pkg_dir
        )

        # 动态库搜索路径
        ld_paths = [
            base_dir,
            os.path.join(base_dir, "_internal"),
            kernel_pkg_dir,
            os.path.join(kernel_pkg_dir, "_libs"),
        ]
        ld_old = env.get("LD_LIBRARY_PATH", "")
        ld_new_parts = [p for p in ld_paths if os.path.exists(p)]
        if ld_new_parts:
            env["LD_LIBRARY_PATH"] = ":".join(
                ld_new_parts + ([ld_old] if ld_old else [])
            )

    # LLM特定优化设置
    env.setdefault("TRITON_DISABLE_JIT", "0")
    env.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
    env.setdefault("DISABLE_TORCHINDUCTOR", "0")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    return env


def launch_tts_service(args):
    """启动TTS服务"""
    logger.info("准备启动TTS服务...")

    if not args.model_dir:
        logger.error("TTS服务需要指定 --model_dir")
        return 1

    if not os.path.exists(args.model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        return 1

    # 准备环境变量
    env = os.environ.copy()
    env = prepare_common_env(env)
    env = prepare_tts_env(env)

    # 构建启动命令
    if getattr(sys, "frozen", False):
        from light_tts.server.api_server import normal_start, parse_args

        new_args = parse_args()

        new_args.mode = args.tts_mode
        new_args.bert_process_num = args.bert_process_num
        new_args.decode_process_num = args.decode_process_num
        new_args.max_total_token_num = args.max_total_token_num
        new_args.max_req_total_len = int(0.8 * args.max_total_token_num)
        new_args.max_req_input_len = int(0.4 * args.max_total_token_num)
        new_args.encode_paral_num = args.encode_paral_num
        new_args.gpt_paral_num = args.gpt_paral_num
        new_args.decode_paral_num = args.decode_paral_num
        normal_start(new_args)

    else:
        cmd = [
            sys.executable,
            "-m",
            "light_tts.server.api_server",
            "--model_dir",
            args.model_dir,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--bert_process_num",
            str(args.bert_process_num),
            "--decode_process_num",
            str(args.decode_process_num),
            "--max_total_token_num",
            str(args.max_total_token_num),
            "--max_req_total_len",
            str(int(args.max_total_token_num * 0.8)),
            "--encode_paral_num",
            str(args.encode_paral_num),
            "--gpt_paral_num",
            str(args.gpt_paral_num),
            "--decode_paral_num",
            str(args.decode_paral_num),
            "--mode",
            args.tts_mode,
        ]

    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")

    try:
        process = subprocess.Popen(cmd, env=env)
        logger.info("TTS服务启动中，按 Ctrl+C 停止服务")
        process.wait()
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭TTS服务...")
        if process:
            process.terminate()
            process.wait()
        logger.info("TTS服务已停止")
    except Exception as e:
        logger.error(f"启动TTS服务失败: {e}")
        return 1

    return 0


def _import_lightllm_module():
    """动态导入lightllm模块，处理命名冲突"""
    try:
        # 在冻结或开发环境，直接使用 lightllm
        if getattr(sys, "frozen", False):
            import lightllm.server.api_server as api_server
            import lightllm.server.api_cli as api_cli
            import lightllm.server.api_start as api_start

            return api_server, api_cli, api_start
        else:
            # 开发环境：直接导入lightllm
            import lightllm.server.api_server as api_server
            import lightllm.server.api_cli as api_cli
            import lightllm.server.api_start as api_start

            return api_server, api_cli, api_start
    except ImportError as e:
        logger.error(f"导入lightllm模块失败: {e}")
        # 如果导入失败，尝试直接调用启动逻辑
        return None, None, None


def launch_llm_service(args):
    """启动LLM服务"""
    logger.info("准备启动LLM服务...")

    if not args.model_dir:
        logger.error("LLM服务需要指定 --model_dir")
        return 1

    if not os.path.exists(args.model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        return 1

    # 准备环境变量
    env = os.environ.copy()
    env = prepare_common_env(env)
    env = prepare_llm_env(env)

    # 尝试导入lightllm模块
    api_server, api_cli, api_start = _import_lightllm_module()
    # try:
    #     import lightllm.server.api_server as api_server
    #     use_internal = True
    # except ImportError:
    #     use_internal = False
    #     logger.warning("无法导入lightllm模块,启动失败")

    server_args = [
        "lightllm.server.api_server",
        "--run_mode",
        args.run_mode,
        "--model_dir",
        args.model_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max_req_total_len",
        str(args.max_req_total_len),
        "--max_total_token_num",
        str(args.max_total_token_num),
        "--cache_capacity",
        str(args.cache_capacity),
        "--mode",
        args.llm_mode,
        "--data_type",
        args.data_type,
        "--mem_fraction",
        str(args.mem_fraction),
        "--tp",
        str(args.tp),
        "--nccl_port",
        str(args.nccl_port),
        "--tokenizer_mode",
        args.tokenizer_mode,
        "--sampling_backend",
        args.sampling_backend,
        "--graph_max_batch_size",
        str(args.graph_max_batch_size),
        "--graph_max_len_in_batch",
        str(args.graph_max_len_in_batch),
        "--chunked_prefill_size",
        str(args.chunked_prefill_size),
        "--use_dynamic_prompt_cache",
    ]

    # 添加LLM特定参数
    if args.quant_type:
        server_args.extend(["--quant_type", args.quant_type])
    if args.trust_remote_code:
        server_args.append("--trust_remote_code")
    if args.enable_concurrent_alloc:
        server_args.append("--enable_concurrent_alloc")
    if args.enable_multimodal:
        server_args.extend(
            [
                "--enable_multimodal",
                "--visual_gpu_ids",
                args.visual_gpu_ids,
                "--visual_nccl_ports",
                str(args.visual_nccl_ports),
                "--visual_infer_batch_size",
                str(args.visual_infer_batch_size),
            ]
        )
    if args.enable_multimodal_audio:
        server_args.extend(
            ["--enable_multimodal_audio", "--audio_gpu_ids", args.audio_gpu_ids]
        )

    old_argv = list(sys.argv)
    sys.argv = server_args

    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info(f"运行模式: {args.run_mode}")
    logger.info(f"推理模式: {args.llm_mode or 'triton_flashdecoding'}")

    try:
        if hasattr(api_server, "main"):
            api_server.main()
        else:
            parser = api_cli.make_argument_parser()
            parsed_args = parser.parse_args()
            parsed_args.mode = args.llm_mode

            if parsed_args.run_mode == "pd_master":
                api_start.pd_master_start(parsed_args)
            elif parsed_args.run_mode == "config_server":
                api_start.config_server_start(parsed_args)
            else:
                api_start.normal_or_p_d_start(parsed_args)
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭LLM服务...")
    finally:
        sys.argv = old_argv


def _is_multiprocessing_child() -> bool:
    """检查是否为多进程子进程"""
    if any(arg.startswith("--multiprocessing") for arg in sys.argv[1:]):
        return True
    if os.environ.get("PYI_CHILD_PROCESS") == "1":
        return True
    if os.environ.get("PYTHONMULTIPROCESSING_SPAWN") == "1":
        return True
    return False


def main():
    """主函数"""
    if _is_multiprocessing_child():
        return 0

    parser = argparse.ArgumentParser(description="统一服务启动器 - 支持TTS和LLM服务")

    # 服务选择
    parser.add_argument(
        "--service", choices=["tts", "llm"], help="选择要启动的服务类型"
    )

    # 通用参数
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument(
        "--port", type=int, default=8080, help="服务器端口 (TTS默认8089, LLM默认8000)"
    )

    # TTS特定参数
    parser.add_argument(
        "--tts_mode", type=str, default="triton_flashdecoding", help="TTS推理模式"
    )
    parser.add_argument("--bert_process_num", type=int, default=1, help="BERT进程数量")
    parser.add_argument(
        "--decode_process_num", type=int, default=1, help="解码进程数量"
    )
    parser.add_argument("--encode_paral_num", type=int, default=50, help="编码并行数量")
    parser.add_argument("--gpt_paral_num", type=int, default=50, help="GPT并行数量")
    parser.add_argument("--decode_paral_num", type=int, default=1, help="解码并行数量")
    parser.add_argument(
        "--max_req_input_len", type=int, default=1024, help="解码并行数量"
    )

    # LLM特定参数
    parser.add_argument(
        "--llm_mode",
        type=str,
        default="triton_flashdecoding",
        help="推理模式，如triton_flashdecoding, ppl_int8kv_flashdecoding等",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="normal",
        help="LLM运行模式 (normal/prefill/decode等)",
    )
    parser.add_argument(
        "--data_type", type=str, default="bf16", help="数据类型 (bf16/fp16/fp8等)"
    )
    parser.add_argument("--quant_type", type=str, help="量化类型")
    parser.add_argument(
        "--max_req_total_len", type=int, default=4000, help="最大请求总长度"
    )
    parser.add_argument(
        "--max_total_token_num", type=int, default=4096, help="最大总token数"
    )
    parser.add_argument("--cache_capacity", type=int, default=1000, help="缓存容量")
    parser.add_argument("--mem_fraction", type=float, default=0.85, help="内存使用比例")
    parser.add_argument("--tp", type=int, default=1, help="张量并行度")
    parser.add_argument("--nccl_port", type=int, default=29500, help="NCCL端口")
    # 多模态配置参数
    parser.add_argument(
        "--enable_multimodal", action="store_true", help="启用多模态支持"
    )
    parser.add_argument(
        "--enable_multimodal_audio", action="store_true", help="启用多模态语音支持"
    )
    parser.add_argument(
        "--visual_gpu_ids", type=str, default="0", help="视觉处理GPU ID"
    )
    parser.add_argument("--audio_gpu_ids", type=str, default="0", help="音频处理GPU ID")
    parser.add_argument(
        "--visual_nccl_ports", type=int, default=29501, help="视觉NCCL端口"
    )
    parser.add_argument(
        "--visual_infer_batch_size", type=int, default=8, help="视觉推理批次大小"
    )

    # 其他配置参数
    parser.add_argument("--tokenizer_mode", type=str, default="auto", help="分词器模式")
    parser.add_argument("--trust_remote_code", action="store_true", help="信任远程代码")
    parser.add_argument(
        "--use_dynamic_prompt_cache", action="store_true", help="使用动态提示缓存"
    )
    parser.add_argument(
        "--sampling_backend", type=str, default="triton_top_kp", help="采样后端"
    )
    parser.add_argument(
        "--enable_concurrent_alloc", action="store_true", help="启用并发分配"
    )
    parser.add_argument(
        "--graph_max_batch_size", type=int, default=4, help="图最大批次大小"
    )
    parser.add_argument(
        "--graph_max_len_in_batch", type=int, default=1024, help="批次中图最大长度"
    )
    parser.add_argument(
        "--chunked_prefill_size", type=int, default=1024, help="分块预填充大小"
    )

    args = parser.parse_args()
    print("DEBUG: args.model_dir =", args.model_dir)

    # 设置默认端口
    if args.port is None:
        args.port = 9001 if args.service == "tts" else 9002

    # 设置多进程启动方式
    try:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # 根据服务类型启动
    if args.service == "tts":
        return launch_tts_service(args)
    else:
        return launch_llm_service(args)


if __name__ == "__main__":
    # 首先，最重要的一步，检查当前是否为子进程
    # if _is_multiprocessing_child():
    #     # 如果是子进程，说明它是由 lightllm 库生成的。
    #     # 它的 sys.argv 包含 lightllm 理解的内部参数。
    #     # 我们必须绕过我们自己的 main() 函数，让 lightllm 的代码来处理。
    #     logger.info("检测到子进程，将控制权交给lightllm。")

    #     # 我们重新导入必要的模块，并运行子进程期望执行的入口点。
    #     try:
    #         from lightllm.server import api_cli, api_start

    #         # lightllm 的参数解析器能正确解析传递给该子进程的内部参数。
    #         parser = api_cli.make_argument_parser()
    #         parsed_args = parser.parse_args()

    #         # 复制启动正确服务组件的逻辑。
    #         if parsed_args.run_mode == "pd_master":
    #             api_start.pd_master_start(parsed_args)
    #         elif parsed_args.run_mode == "config_server":
    #             api_start.config_server_start(parsed_args)
    #         else:
    #             api_start.normal_or_p_d_start(parsed_args)
    #     except Exception as e:
    #         logger.error(f"子进程出错: {e}")
    #         sys.exit(1)

    # else:
    #     # 如果不是子进程，那么这就是面向用户的主执行过程。
    #     # 运行标准的设置和参数解析。
    try:
        import multiprocessing as _mp

        _mp.freeze_support()
    except Exception:
        pass

    # 现在，调用你原来的 main 函数。
    sys.exit(main())
