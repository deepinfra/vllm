import asyncio
import io
import time

import aiohttp
import requests
import codecs
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Optional, List, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.async_llava_engine import AsyncLLaVAEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from io import BytesIO
from PIL import Image
import base64


logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine: Union[AsyncLLMEngine, AsyncLLaVAEngine],
        served_model: str,
        response_role: str,
        lora_modules: Optional[List[LoRA]] = None,
        chat_template=None,
        model_type=None,
    ):
        super().__init__(engine=engine, served_model=served_model, lora_modules=lora_modules)
        self.response_role = response_role
        self._load_chat_template(chat_template)
        self.model_type = model_type

    async def read_image(self, request_id, session, url: str):
        if url.startswith("http"):
            return await self.download_image(request_id, session, url)
        else:
            data = url.split(",")[1] if url.startswith("data:") else url
            try:
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
                return image
            except Exception as e:
                logger.error(f"Error decoding base64 image data: {e} [request_id={request_id}]")
                return None

    async def download_image(self, request_id, session, url):
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60, connect=10)) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type')
                        content_length = response.headers.get('Content-Length')
                        if content_type in ['image/jpeg', 'image/png', 'image/webp'] and content_length and int(content_length) < 20 * 1024 * 1024:
                            image_data = await response.read()
                            try:
                                image = Image.open(io.BytesIO(image_data))
                                logger.info(f"Image downloaded successfully [request_id={request_id}]")
                                return image
                            except (IOError, SyntaxError) as e:
                                logger.error(f"Error opening image: {e} [request_id={request_id}]")
                                break  # Don't retry if there's an error opening the image
                        else:
                            logger.warning(f"Skipped image download, invalid content type or size [request_id={request_id}]")
                    else:
                        logger.warning(f"Failed to download image, status code: {response.status} [request_id={request_id}]")
                        break  # Don't retry for non-200 status codes
            except aiohttp.ClientError as e:
                logger.error(f"Failed to download image, error: {e} [request_id={request_id}]")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading image [request_id={request_id}]")
            logger.info(f"Retry {attempt+1} [request_id={request_id}]")
            await asyncio.sleep(1)  # Wait for a moment before retrying
        logger.error(f"Failed to download image after retries [request_id={request_id}]")
        return None

    async def fetch_images(self, request_id, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.read_image(request_id, session, url) for url in urls]
            return await asyncio.gather(*tasks)


    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.

        NOTE: Currently we do not support the following features:
            - function_call (Users should implement this by themselves)
            - logit_bias (to be supported by vLLM engine)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if request.logit_bias is not None and len(request.logit_bias) > 0:
            # TODO: support logit_bias in vLLM engine.
            return self.create_error_response(
                "logit_bias is not currently supported")

        request_id = f"cmpl-{random_uuid()}"

        chat_config = request.chat_config if request.chat_config else {}
        prompt = ""
        images = []
        try:
            image_urls = []
            if self.model_type == "vision":
                for message in request.messages:
                    if message["role"] == "user":
                        prompt += chat_config.get("user_prefix",
                                                  "USER:\n")  # "USER:\n"
                        if isinstance(message["content"], str):
                            prompt += f"{message['content']}\n"
                        else:
                            for content in message["content"]:
                                if content["type"] == "text":
                                    prompt += f"{content['text']}\n"
                                if content["type"] == "image_url":
                                    # read the image
                                    url = content["image_url"]["url"]
                                    image_urls.append(url)
                                    prompt += chat_config.get(
                                        "image_token", "<image>\n")
                    if message["role"] == "assistant":
                        prompt += chat_config.get(
                            "assistant_prefix",
                            "ASSISTANT:\n")
                        prompt += f"{message['content']}\n"

                prompt += chat_config.get("assistant_prefix", "ASSISTANT:\n")
                assert prompt.count(chat_config.get("image_token", "<image>\n")) == len(image_urls), \
                    "Number of images and image tokens should be same"
                images = await self.fetch_images(request_id, image_urls)
                assert len([x for x in images if x]) == len(image_urls), \
                    "Number of images fetched should be same as number of image urls"
            else:
                prompt = self.tokenizer.apply_chat_template(
                    conversation=request.messages,
                    tokenize=False,
                    add_generation_prompt=request.add_generation_prompt)

        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))


        try:
            # if not self.model_type == "vision":
            #     token_ids = None
            # else:
            token_ids = self._validate_prompt_and_tokenize(request,
                                                            prompt=prompt)
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = self.engine.generate(prompt, sampling_params,
                                                request_id, token_ids, lora_request, images=images)
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id)
        else:
            return await self.chat_completion_full_generator(
                request, raw_request, result_generator, request_id)

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1].role

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

        model_name = request.model
        created_time = int(time.monotonic())
        chunk_object_type = "chat.completion.chunk"

        # Send first response for each request.n (index) with the role
        role = self.get_chat_request_role(request)
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None)
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 object=chunk_object_type,
                                                 created=created_time,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index

                if finish_reason_sent[i]:
                    continue

                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                else:
                    # Send the finish response for each request.n only once
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=previous_num_tokens[i],
                        total_tokens=prompt_tokens + previous_num_tokens[i],
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        finish_reason=output.finish_reason)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    if final_usage is not None:
                        chunk.usage = final_usage
                    data = chunk.model_dump_json(exclude_unset=True,
                                                 exclude_none=True)
                    yield f"data: {data}\n\n"
                    finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
            self, request: ChatCompletionRequest, raw_request: Request,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = request.model
        created_time = int(time.monotonic())
        final_res: RequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    def _load_chat_template(self, chat_template):
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                self.tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info(
                f"Using supplied chat template:\n{self.tokenizer.chat_template}"
            )
        elif self.tokenizer.chat_template is not None:
            logger.info(
                f"Using default chat template:\n{self.tokenizer.chat_template}"
            )
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")
