import asyncio
import os
import torch
from typing import List, Tuple, Union

from lightllm.lightllm.server.api_cli import make_argument_parser
from lightllm.lightllm.server.api_start import normal_or_p_d_start
from lightllm.lightllm.server.api_http import g_objs
from lightllm.lightllm.server.api_http import SamplingParams as LightllmSamplingParams
from lightllm.lightllm.server.api_http import MultimodalParams as LightllmMultimodalParams
from lightllm.lightllm.server.core.objs.req import FinishStatus
from lightllm.lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class BatchEngine():
    """
    The engine is patched to support batch multi-modal generate, and early image preprocessing.
    """

    def __init__(
        self,
        model_path: str,
        **kwargs,
    ):
        torch.multiprocessing.set_start_method("spawn")
        self.model_path = model_path
        self.args = self._parse_start_args()

    def _parse_start_args(self):
        parser = make_argument_parser()
        args = parser.parse_args()
        args.run_mode = "normal"
        args.model_dir = self.model_path
        return args

    def init_engine(self):
        normal_or_p_d_start(self.args)
        self.http_server_manager = g_objs.httpserver_manager

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
