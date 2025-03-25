# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import time
from collections.abc import AsyncGenerator
from math import ceil
from typing import Final, Optional, Union, cast

from fastapi import Request
from fastapi.responses import StreamingResponse

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    DeltaMessage, ErrorResponse, RequestResponseMetadata, TranscriptionRequest,
    TranscriptionResponse, TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse, UsageInfo, SpeechRequest)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils import PlaceholderModule


logger = init_logger(__name__)


class OpenAIServingSpeech(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params)

    def format_prompt(self, request: SpeechRequest) -> str:
        adapted_prompt = f"{request.voice}: {request.input}"
        prompt_tokens = self.engine_client.get_tokenizer()(adapted_prompt, return_tensors="pt")

    async def create_speech(
        self, request: SpeechRequest, raw_request: Request
    ) -> Union[StreamingResponse, ErrorResponse]:
        request_id = f"trsc-{self._base_request_id(raw_request)}"

        logger.info(f"Request id: {request_id} input: {request.input} model: {request.model} voice: {request.voice} response_format: {request.response_format}")

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        audio_buffer = io.BytesIO()
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=orpheus_speech.wav"}
        )

