import os
import time

from base64 import b64encode
from pathlib import Path
from typing import AsyncIterable, Iterable, List, Optional, Union

from lightllm.server.api_http import SamplingParams as LightllmSamplingParams
from lightllm.server.api_http import MultimodalParams as LightllmMultimodalParams
from lightllm.utils.log_utils import init_logger

from ...model.vlm_lightllm_model.engine import BatchEngine
from .base_predictor import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    BasePredictor,
)


logger = init_logger(__name__)

class LightllmEnginePredictor(BasePredictor):
    def __init__(
        self,
        model_path: str,
        #
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        self.model_path = model_path

        self.engine = BatchEngine(
            model_path=model_path,
        )
        self.engine.init_engine()
        self.tokenizer = self.engine.get_tokenizer()

    def _build_sampling_params(
        self,
        sampling_params: dict,
    ) -> LightllmSamplingParams:
        sampling_params = LightllmSamplingParams()
        sampling_params.init(
            tokenizer=self.tokenizer,
            **sampling_params,
        )
        return sampling_params

    def _build_multimodal_params(
        self,
        multimodal_params: dict,
    ) -> LightllmMultimodalParams:
        multimodal_params = LightllmMultimodalParams(**multimodal_params)
        return multimodal_params

    def predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )[0]

    def batch_predict(
        self,
        images: List[str] | List[bytes],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:

        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        prompts = [self.build_prompt(prompt) for prompt in prompts]

        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        if presence_penalty is None:
            presence_penalty = self.presence_penalty
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = self.no_repeat_ngram_size
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # see SamplingParams for more details
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "custom_params": {
                "no_repeat_ngram_size": no_repeat_ngram_size,
            },
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        }
        lightllm_sampling_params = self._build_sampling_params(sampling_params)
        lightllm_multimodal_params = self._build_multimodal_params(sampling_params.get("multimodal_params", {}))

        output = self.engine.generate(
            prompt=prompts,
            sampling_params=lightllm_sampling_params,
            multimodal_params=lightllm_multimodal_params,
        )
        return [item["text"] for item in output]

    def stream_predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        raise NotImplementedError("Streaming is not supported yet.")

    async def aio_predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        output = await self.aio_batch_predict(
            [image],  # type: ignore
            [prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        return output[0]

    async def aio_batch_predict(
        self,
        images: List[str] | List[bytes],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:

        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        prompts = [self.build_prompt(prompt) for prompt in prompts]

        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        if presence_penalty is None:
            presence_penalty = self.presence_penalty
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = self.no_repeat_ngram_size
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # see SamplingParams for more details
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "custom_params": {
                "no_repeat_ngram_size": no_repeat_ngram_size,
            },
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        }
        lightllm_sampling_params = self._build_sampling_params(sampling_params)
        lightllm_multimodal_params = self._build_multimodal_params(sampling_params.get("multimodal_params", {}))

        output = await self.engine.async_generate(
            prompt=prompts,
            sampling_params=lightllm_sampling_params,
            multimodal_params=lightllm_multimodal_params,
        )
        ret = []
        for item in output:  # type: ignore
            ret.append(item["text"])
        return ret

    async def aio_stream_predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> AsyncIterable[str]:
        raise NotImplementedError("Streaming is not supported yet.")

    def close(self):
        self.engine.shutdown()
