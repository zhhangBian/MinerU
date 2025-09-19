import asyncio
import os
import sys
import torch
import threading
from typing import List, Tuple, Union

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import normal_or_p_d_start
from lightllm.server.api_http import SamplingParams as LightllmSamplingParams
from lightllm.server.api_http import MultimodalParams as LightllmMultimodalParams
from lightllm.server.core.objs.req import FinishStatus
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class BatchEngine():
    def __init__(
        self,
        model_path: str,
        **kwargs,
    ):
        torch.multiprocessing.set_start_method("spawn")
        self.model_path = model_path
        self.args = self._parse_start_args()
        self.http_server_manager = None
        self._server_started = threading.Event()
        self._server_thread = None

    def _parse_start_args(self):
        """解析lightllm启动参数，处理以lightllm_开头的命令行参数"""
        # 获取原始命令行参数
        original_argv = sys.argv[1:]  # 去掉脚本名

        # 筛选并处理lightllm参数
        lightllm_args = []
        i = 0
        while i < len(original_argv):
            arg = original_argv[i]

            if arg.startswith('--lightllm_'):
                # 去掉lightllm_前缀，保留--
                new_arg = '--' + arg[11:]  # 去掉'--lightllm_'，保留'--'
                lightllm_args.append(new_arg)

                # 处理参数值（可能有多个值，如列表型参数）
                j = i + 1
                while j < len(original_argv) and not original_argv[j].startswith('--'):
                    lightllm_args.append(original_argv[j])
                    j += 1
                i = j - 1  # 调整索引到最后一个处理的参数
            i += 1

        # 记录处理的lightllm参数
        if lightllm_args:
            logger.info(f"Processed lightllm arguments: {lightllm_args}")

        # 创建lightllm的parser并解析处理后的参数
        parser = make_argument_parser()
        args = parser.parse_args(lightllm_args)

        # 保持原有的固定设置
        args.run_mode = "normal"
        args.model_dir = self.model_path

        return args

    def _start_server_thread(self):
        """在后台线程中启动lightllm服务器"""
        try:
            logger.info("Starting lightllm server in background thread...")
            normal_or_p_d_start(self.args)
            self._server_started.set()
            logger.info("Lightllm server started successfully")
        except Exception as e:
            logger.error(f"Failed to start lightllm server: {e}")
            raise

    def init_engine(self):
        """非阻塞方式初始化引擎"""
        # 在后台线程中启动服务器
        self._server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
        self._server_thread.start()

        # 等待服务器启动完成（带超时）
        logger.info("Waiting for lightllm server to start...")
        if not self._server_started.wait(timeout=120):  # 2分钟超时
            raise RuntimeError("Lightllm server failed to start within timeout")

        # 导入全局对象
        try:
            from lightllm.server.api_http import g_objs
            self.http_server_manager = g_objs.httpserver_manager
            logger.info("Successfully connected to lightllm server")
        except Exception as e:
            logger.error(f"Failed to import g_objs: {e}")
            raise

    def is_ready(self):
        """检查服务是否已准备就绪"""
        return self._server_started.is_set() and self.http_server_manager is not None

    def get_tokenizer(self):
        return self.http_server_manager.tokenizer

    def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: LightllmSamplingParams,
        multimodal_params: LightllmMultimodalParams,
        is_health_req: bool = False,
    ) -> Tuple[int, str, dict, FinishStatus]:
        return self.http_server_manager.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=None,
            is_health_req=is_health_req,
        )

    async def async_generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: LightllmSamplingParams,
        multimodal_params: LightllmMultimodalParams,
        is_health_req: bool = False,
    ) -> Tuple[int, str, dict, FinishStatus]:
        return await self.http_server_manager.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=None,
            is_health_req=is_health_req,
        )

    async def shutdown(self):
        logger.info("Received signal to shutdown. Performing graceful shutdown...")
        asyncio.sleep(3)

        # 杀掉所有子进程
        import psutil
        import signal

        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGKILL)
        logger.info("Graceful shutdown completed.")
        return
