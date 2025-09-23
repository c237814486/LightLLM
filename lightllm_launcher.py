#!/usr/bin/env python3
"""
LightLLM 全环境自包含启动入口
仅需用户提供关键参数，其余环境与依赖均随可执行程序打包。
解决与TTS框架的lightllm命名冲突问题。
"""

import os
import sys
import argparse
import logging
import warnings
import importlib.util

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 避免 typeguard 在冻结环境中尝试读取源码导致失败
os.environ.setdefault("TYPEGUARD_DISABLE", "true")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lightllm_launcher")


def _get_exe_base_dir() -> str:
    """返回冻结/开发环境下的基准目录。

    - ONEFILE: sys._MEIPASS（临时解压目录）
    - ONEDIR:  可执行文件所在目录
    - 开发环境: 本文件所在目录
    """
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


def _prepare_runtime_env(env: dict) -> dict:
    """准备运行时环境变量"""
    base_dir = _get_exe_base_dir()

    # 设置Python路径，将 lightllm 添加到路径中
    lightllm_path = os.path.join(base_dir, "lightllm")
    if os.path.exists(lightllm_path):
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{lightllm_path}:{current_pythonpath}"
            if current_pythonpath
            else lightllm_path
        )

    # 配置 lightllm_kernel 路径（PYTHONPATH 与动态库路径）
    # 在冻结环境中，lightllm_kernel 在 _internal 目录下
    if getattr(sys, "frozen", False):
        kernel_pkg_dir = os.path.join(base_dir, "_internal", "lightllm_kernel")
    else:
        kernel_pkg_dir = os.path.join(base_dir, "lightllm_kernel")

    if os.path.exists(kernel_pkg_dir):
        # 确保 python 可以找到 lightllm_kernel 包
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{kernel_pkg_dir}:{current_pythonpath}"
            if current_pythonpath
            else kernel_pkg_dir
        )

        # 动态库搜索路径：顶层、_internal、包目录、_libs 目录
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

    # 抑制常见警告
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"

    # 设置CUDA相关环境变量
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # 禁用一些可能导致问题的优化
    env.setdefault("TRITON_DISABLE_JIT", "0")  # 保持triton JIT，但设置合理的限制
    env.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
    env.setdefault("DISABLE_TORCHINDUCTOR", "0")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # 设置多进程相关
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    return env


def _is_multiprocessing_child() -> bool:
    """检查是否为多进程子进程"""
    if any(arg.startswith("--multiprocessing") for arg in sys.argv[1:]):
        return True
    if os.environ.get("PYI_CHILD_PROCESS") == "1":
        return True
    if os.environ.get("PYTHONMULTIPROCESSING_SPAWN") == "1":
        return True
    return False


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


def _start_with_subprocess(args):
    """使用subprocess方式启动LightLLM服务"""
    import subprocess

    # 构建启动命令
    cmd = [
        sys.executable,
        "-m",
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
        args.mode,
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
    ]

    # 添加可选参数
    if args.quant_type:
        cmd.extend(["--quant_type", args.quant_type])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.use_dynamic_prompt_cache:
        cmd.append("--use_dynamic_prompt_cache")
    if args.enable_concurrent_alloc:
        cmd.append("--enable_concurrent_alloc")
    if args.enable_multimodal:
        cmd.extend(
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
        cmd.extend(["--enable_multimodal_audio", "--audio_gpu_ids", args.audio_gpu_ids])

    logger.info("正在启动 LightLLM 服务(subprocess模式)...")
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info(f"运行模式: {args.run_mode}")
    logger.info(f"推理模式: {args.mode}")
    logger.info(f"数据类型: {args.data_type}")
    if args.quant_type:
        logger.info(f"量化类型: {args.quant_type}")

    try:
        # 启动子进程
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"子进程启动失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务器...")
        return 0


def main() -> int:
    """主函数"""
    # 子进程不执行顶层参数解析
    if _is_multiprocessing_child():
        return 0

    parser = argparse.ArgumentParser(description="LightLLM 全环境自包含启动器")

    # 必需参数
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")

    # 服务器配置参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="normal",
        choices=["normal", "prefill", "decode", "pd_master", "config_server"],
        help="运行模式",
    )

    # 模型配置参数
    parser.add_argument(
        "--mode",
        type=str,
        default="triton_flashdecoding",
        help="推理模式，如triton_flashdecoding, ppl_int8kv_flashdecoding等",
    )
    parser.add_argument(
        "--data_type", type=str, default="bf16", help="数据类型，如bf16, fp16, fp8等"
    )
    parser.add_argument(
        "--quant_type", type=str, default=None, help="量化类型，如vllm-fp8w8a8等"
    )

    # 性能配置参数
    parser.add_argument(
        "--max_req_total_len", type=int, default=4000, help="最大请求总长度"
    )
    parser.add_argument(
        "--max_total_token_num", type=int, default=4096, help="最大总token数"
    )
    parser.add_argument("--cache_capacity", type=int, default=20000, help="缓存容量")
    parser.add_argument("--mem_fraction", type=float, default=0.9, help="内存使用比例")

    # 分布式配置参数
    parser.add_argument("--tp", type=int, default=1, help="张量并行度")
    parser.add_argument("--nccl_port", type=int, default=28765, help="NCCL端口")

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

    if not os.path.exists(args.model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        return 1

    # 准备运行环境
    _prepare_runtime_env(os.environ)

    try:
        # 设置多进程启动方式
        try:
            import torch.multiprocessing as mp

            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

        # 动态导入lightllm模块
        api_server, api_cli, api_start = _import_lightllm_module()

        if api_server is None:
            # 如果导入失败，使用subprocess方式启动
            logger.info("使用subprocess方式启动LightLLM服务...")
            return _start_with_subprocess(args)

        # 构建参数列表
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
            args.mode,
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
        ]

        # 添加可选参数
        if args.quant_type:
            server_args.extend(["--quant_type", args.quant_type])
        if args.trust_remote_code:
            server_args.append("--trust_remote_code")
        if args.use_dynamic_prompt_cache:
            server_args.append("--use_dynamic_prompt_cache")
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

        # 设置sys.argv并启动服务器
        old_argv = list(sys.argv)
        sys.argv = server_args

        logger.info("正在启动 LightLLM 服务(自包含环境)...")
        logger.info(f"模型目录: {args.model_dir}")
        logger.info(f"服务地址: http://{args.host}:{args.port}")
        logger.info(f"运行模式: {args.run_mode}")
        logger.info(f"推理模式: {args.mode}")
        logger.info(f"数据类型: {args.data_type}")
        if args.quant_type:
            logger.info(f"量化类型: {args.quant_type}")

        # 调用lightllm的main函数
        if hasattr(api_server, "main"):
            api_server.main()
        else:
            # 如果没有main函数，直接调用启动逻辑
            parser = api_cli.make_argument_parser()
            parsed_args = parser.parse_args()

            if parsed_args.run_mode == "pd_master":
                api_start.pd_master_start(parsed_args)
            elif parsed_args.run_mode == "config_server":
                api_start.config_server_start(parsed_args)
            else:
                api_start.normal_or_p_d_start(parsed_args)

        sys.argv = old_argv
        return 0

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务器...")
        return 0
    except Exception as e:
        logger.exception(f"启动失败: {e}")
        return 1


if __name__ == "__main__":
    try:
        import multiprocessing as _mp

        _mp.freeze_support()
    except Exception:
        pass
    sys.exit(main())
