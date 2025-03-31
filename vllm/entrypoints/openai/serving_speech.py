# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import os
import json
import time

import numpy as np
import struct
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import torch

from typing import Final, Optional, Union, cast

from fastapi import Request
from fastapi.responses import StreamingResponse

from snac import SNAC

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse, RequestResponseMetadata, SpeechRequest, CompletionRequest, CompletionResponse)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

thread_pool = ThreadPoolExecutor(max_workers=16)

TEMPERATURE = 0.4
TOP_P = 0.9
MAX_TOKENS = 2000
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000
MEDIA_TYPE_INFO = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

def convert_to_audio(snac_model, multiframe: list[int]) -> Optional[bytes]:
    st = time.monotonic()
    frames = []
    if len(multiframe) < 7:
        return None

    # codes_0 = torch.tensor([], device="cpu", dtype=torch.int32)
    # codes_1 = torch.tensor([], device="cpu", dtype=torch.int32)
    # codes_2 = torch.tensor([], device="cpu", dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    # for j in range(num_frames):
    #     i = 7 * j
    #     if codes_0.shape[0] == 0:
    #         codes_0 = torch.tensor([frame[i]], device="cpu", dtype=torch.int32)
    #     else:
    #         codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device="cpu", dtype=torch.int32)])
    #
    #     if codes_1.shape[0] == 0:
    #
    #         codes_1 = torch.tensor([frame[i + 1]], device="cpu", dtype=torch.int32)
    #         codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device="cpu", dtype=torch.int32)])
    #     else:
    #         codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 1]], device="cpu", dtype=torch.int32)])
    #         codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device="cpu", dtype=torch.int32)])
    #
    #     if codes_2.shape[0] == 0:
    #         codes_2 = torch.tensor([frame[i + 2]], device="cpu", dtype=torch.int32)
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device="cpu", dtype=torch.int32)])
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device="cpu", dtype=torch.int32)])
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device="cpu", dtype=torch.int32)])
    #     else:
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 2]], device="cpu", dtype=torch.int32)])
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device="cpu", dtype=torch.int32)])
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device="cpu", dtype=torch.int32)])
    #         codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device="cpu", dtype=torch.int32)])

    codes_0, codes_1, codes_2 = [], [], []
    for j in range(num_frames):
        i = 7 * j
        codes_0.append(frame[i])
        codes_1 += [frame[i + 1], frame[i + 4]]
        codes_2 += [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]]

    codes = [
        torch.tensor([codes_0], dtype=torch.int32, device="cuda"),
        torch.tensor([codes_1], dtype=torch.int32, device="cuda"),
        torch.tensor([codes_2], dtype=torch.int32, device="cuda"),
    ]

    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(
            codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return

    st = time.monotonic()
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)

    st = time.monotonic()
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    return audio_bytes


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
        st = time.monotonic()
        logger.info(f"[{time.monotonic() - st:.3f} sec] TEMIRULAN OpenAIServingSpeech init started")
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)
        self.serving_completion = OpenAIServingCompletion(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids)
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        logger.info(f"[{time.monotonic() - st:.3f} sec] TEMIRULAN OpenAIServingCompletion finished")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.snac_device = "cuda"
        self.snac_model.to(self.snac_device)
        logger.info(f"[{time.monotonic() - st:.3f} sec] TEMIRULAN snac model initialized and moved to {self.snac_device}")

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params)

        logger.info(f"[{time.monotonic() - st:.3f} sec] TEMIRULAN OpenAIServingSpeech init finished")
        self.request_started_time = {}

    async def format_prompt(self, request: SpeechRequest, tokenizer: AnyTokenizer) -> str:
        adapted_prompt = f"{request.voice}: {request.input}"
        prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        return tokenizer.decode(all_input_ids[0])

    def turn_token_into_id(self, token_string: str, index: int):
        token_string = token_string.strip()
        last_token_start = token_string.rfind("<custom_token_")
        if last_token_start == -1:
            return None
        last_token = token_string[last_token_start:]
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except ValueError:
                return None
        else:
            return None

    async def create_speech_stream(self, request_id: str, completion_generator: AsyncGenerator[str, None] | list[str]) -> AsyncGenerator[bytes, None]:
        st = time.monotonic()
        #logger.info(f"[{time.monotonic() - self.request_started_time.get(request_id, -1):.3f} sec] TEMIRULAN r_id:{request_id} OpenAIServingCompletion finished")
        buffer = ""
        token_buffer = []
        token_count = 0
        audio_chunk_count = 0
        convert_audio_time_sec = 0

        async def async_wrap(iterable):
            for item in iterable:
                yield item
        if isinstance(completion_generator, list):
            completion_generator = async_wrap(completion_generator)

        async for chunk in completion_generator:
            data = chunk[len("data: "):].strip()
            if data != "[DONE]":
                data = json.loads(data)
                buffer += data['choices'][0]['text']
                while "><custom_token_" in buffer:
                    token_end = buffer.find(">", buffer.find("_token_")) + 1
                    if token_end > 0:
                        complete_token = "<custom" + buffer[buffer.find("_token_"):token_end]
                        token_id = self.turn_token_into_id(complete_token, token_count)
                        if token_id is not None and token_id > 0:
                            token_buffer.append(token_id)
                            token_count += 1
                            if token_count % 7 == 0 and token_count > 27:
                                buffer_to_proc = token_buffer[-28:]
                                _st = time.monotonic()
                                loop = asyncio.get_running_loop()
                                audio_samples = await loop.run_in_executor(thread_pool, convert_to_audio, self.snac_model, buffer_to_proc)
                                #audio_samples = await asyncio.to_thread(convert_to_audio, buffer_to_proc)
                                _en = time.monotonic()
                                logger.info(f"[{time.monotonic() - self.request_started_time.get(request_id, -1):.3f} sec] TEMIRULAN r_id:{request_id} single audio convertion finished in {_en - _st:.2f} sec")
                                convert_audio_time_sec += _en - _st
                                audio_chunk_count += 1
                                if audio_samples is not None:
                                    yield audio_samples
                        buffer = buffer[token_end:]
        logger.info(f"[{time.monotonic() - self.request_started_time.get(request_id, -1):.3f} sec] TEMIRULAN r_id:{request_id} finished create_speech_stream "
                    f"convert audio completed in {convert_audio_time_sec:.2f} sec, create speech stream finished in {time.monotonic() - st:.2f} sec, token_count: {token_count}, total audio_chunks: {audio_chunk_count}")
        #logger.info(f"Request id: {request_id} finished generating, total number of tokens: {token_count}, total audio chunks: {audio_chunk_count}")


    async def create_speech(
        self, request: SpeechRequest, raw_request: Request
    ) -> Union[StreamingResponse, ErrorResponse, CompletionResponse]:
        request_id = f"cspc-{self._base_request_id(raw_request)}"
        tokenizer = (await self.engine_client.get_tokenizer())

        logger.info(f"Received request id: {request_id} request: {request.to_str()}")
        self.request_started_time[request_id] = time.monotonic()
        #logger.info(f"[{time.monotonic() - self.request_started_time.get(request_id, -1):.3f} sec] TEMIRULAN r_id:{request_id} started create speech")

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        formatted_prompt = await self.format_prompt(request, tokenizer)

        #logger.info(f"[{time.monotonic() - self.request_started_time.get(request_id, -1):.3f} sec] TEMIRULAN r_id:{request_id} format prompt finished")

        completion_request = CompletionRequest(
            model=request.model,
            prompt=formatted_prompt,
            stream=True,
            max_tokens=request.max_tokens or MAX_TOKENS,
            temperature=request.temperature or TEMPERATURE,
            repetition_penalty=request.repetition_penalty or REPETITION_PENALTY,
            top_p=request.top_p or TOP_P,
            stop=["<|endoftext|>"],
        )

        stream_generator = await self.serving_completion.create_completion(completion_request, raw_request)

        #logger.info(
        #    f"[{time.monotonic() - self.request_started_time.get(request_id, -1):.3f} sec] TEMIRULAN r_id:{request_id} finished calling completion stream request, type: {type(stream_generator)}")

        media_type = MEDIA_TYPE_INFO.get(request.response_format, "audio/wav")
        content_disposition = f"attachment; filename=orpheus_speech.{request.response_format}"
        return StreamingResponse(
            self.create_speech_stream(request_id, stream_generator),
            media_type=media_type,
            headers={"Content-Disposition": content_disposition},
        )

