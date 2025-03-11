# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import numpy as np
import re
import time
from collections.abc import AsyncGenerator
from math import ceil
from pydub import AudioSegment
from typing import Final, Optional, Union, cast

from fastapi import Request

from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    DeltaMessage, ErrorResponse, RequestResponseMetadata, TranscriptionRequest,
    TranscriptionResponse, TranscriptionResponseVerbose, TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.processor import cached_get_processor
from vllm.transformers_utils.tokenizer import decode_tokens
from vllm.utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

logger = init_logger(__name__)

# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages#supported-languages
# TODO these configs should live somewhere with the model so we can support
# additional ones

ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh"
}
ISO639_1_OTHER_LANGS = {
    "lo": "Lao",
    "jw": "Javanese",
    "tk": "Turkmen",
    "yi": "Yiddish",
    "so": "Somali",
    "bn": "Bengali",
    "nn": "Norwegian Nynorsk",
    "si": "Sinhala",
    "yo": "Yoruba",
    "sa": "Sanskrit",
    "mi": "Māori",
    "fo": "Faroese",  # codespell:ignore
    "mt": "Maltese",
    "tg": "Tajik",
    "mg": "Malagasy",
    "haw": "Hawaiian",
    "km": "Khmer",
    "br": "Breton",
    "ps": "Pashto",
    "ln": "Lingala",
    "la": "Latin",
    "ml": "Malayalam",
    "sq": "Albanian",
    "su": "Sundanese",
    "eu": "Basque",
    "ka": "Georgian",
    "uz": "Uzbek",
    "sn": "Shona",
    "ht": "Haitian",
    "as": "Assamese",
    "mn": "Mongolian",
    "te": "Telugu",
    "pa": "Panjabi",
    "tt": "Tatar",
    "gu": "Gujarati",
    "oc": "Occitan",
    "ha": "Hausa",
    "ba": "Bashkir",
    "my": "Burmese",
    "sd": "Sindhi",
    "am": "Amharic",
    "lb": "Luxembourgish",
    "bo": "Tibetan"
}
LANG_ID_TO_LANG_TOKEN = {
    "whisper-v3": {
        50259: "<|en|>",
        50260: "<|zh|>",
        50261: "<|de|>",
        50262: "<|es|>",
        50263: "<|ru|>",
        50264: "<|ko|>",
        50265: "<|fr|>",
        50266: "<|ja|>",
        50267: "<|pt|>",
        50268: "<|tr|>",
        50269: "<|pl|>",
        50270: "<|ca|>",
        50271: "<|nl|>",
        50272: "<|ar|>",
        50273: "<|sv|>",
        50274: "<|it|>",
        50275: "<|id|>",
        50276: "<|hi|>",
        50277: "<|fi|>",
        50278: "<|vi|>",
        50279: "<|he|>",
        50280: "<|uk|>",
        50281: "<|el|>",
        50282: "<|ms|>",
        50283: "<|cs|>",
        50284: "<|ro|>",
        50285: "<|da|>",
        50286: "<|hu|>",
        50287: "<|ta|>",
        50288: "<|no|>",
        50289: "<|th|>",
        50290: "<|ur|>",
        50291: "<|hr|>",
        50292: "<|bg|>",
        50293: "<|lt|>",
        50294: "<|la|>",
        50295: "<|mi|>",
        50296: "<|ml|>",
        50297: "<|cy|>",
        50298: "<|sk|>",
        50299: "<|te|>",
        50300: "<|fa|>",
        50301: "<|lv|>",
        50302: "<|bn|>",
        50303: "<|sr|>",
        50304: "<|az|>",
        50305: "<|sl|>",
        50306: "<|kn|>",
        50307: "<|et|>",
        50308: "<|mk|>",
        50309: "<|br|>",
        50310: "<|eu|>",
        50311: "<|is|>",
        50312: "<|hy|>",
        50313: "<|ne|>",
        50314: "<|mn|>",
        50315: "<|bs|>",
        50316: "<|kk|>",
        50317: "<|sq|>",
        50318: "<|sw|>",
        50319: "<|gl|>",
        50320: "<|mr|>",
        50321: "<|pa|>",
        50322: "<|si|>",
        50323: "<|km|>",
        50324: "<|sn|>",
        50325: "<|yo|>",
        50326: "<|so|>",
        50327: "<|af|>",
        50328: "<|oc|>",
        50329: "<|ka|>",
        50330: "<|be|>",
        50331: "<|tg|>",
        50332: "<|sd|>",
        50333: "<|gu|>",
        50334: "<|am|>",
        50335: "<|yi|>",
        50336: "<|lo|>",
        50337: "<|uz|>",
        50338: "<|fo|>",
        50339: "<|ht|>",
        50340: "<|ps|>",
        50341: "<|tk|>",
        50342: "<|nn|>",
        50343: "<|mt|>",
        50344: "<|sa|>",
        50345: "<|lb|>",
        50346: "<|my|>",
        50347: "<|bo|>",
        50348: "<|tl|>",
        50349: "<|mg|>",
        50350: "<|as|>",
        50351: "<|tt|>",
        50352: "<|haw|>",
        50353: "<|ln|>",
        50354: "<|ha|>",
        50355: "<|ba|>",
        50356: "<|jw|>",
        50357: "<|su|>",
        50358: "<|yue|>",
    },
}

# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
MAX_AUDIO_CLIP_FILESIZE_MB = 25


def trim_lang_token(lang_token: str) -> str:
    return lang_token.strip().replace("<|", "").replace("|>", "")


# ONLY FOR Whisper v3
def is_timestamp_token_id(token_id: int) -> bool:
    return 50365 <= token_id <= 51865


def timestamp_token_id_to_timestamp(token_id: int) -> float:

    return (token_id - 50365) * 0.02


def remove_timestamps(text: str) -> str:
    pattern = r'<\|\s*[+-]?\d*\.?\d+\s*\|>'
    return re.sub(pattern, '', text)


def load_with_librosa(audio_data: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_data) as audio_bytes:
        return librosa.load(audio_bytes, sr=None)


def load_with_audio_segment(audio_data: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(audio_data) as audio_stream:
        audio_segment = AudioSegment.from_file(audio_stream)
    with io.BytesIO() as wav_io:
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        return librosa.load(wav_io, sr=None)


def load_audio(audio_data: bytes, request_id: str) -> tuple[np.ndarray | None, int | None]:
    logger.info(f"{request_id} - Starting audio loading into numpy.")
    loaders = [
        ("librosa", load_with_librosa),
        ("audio_segment", load_with_audio_segment),
    ]
    for name, loader in loaders:
        try:
            result = loader(audio_data)
            logger.info(f"{request_id} - Audio loaded successfully using {name}.")
            return result
        except Exception as e:
            logger.warning(f"{request_id} - {name} loader failed: {e}")

    logger.error(f"{request_id} - All audio loading methods failed.")
    return None, None


class OpenAIServingTranscription(OpenAIServing):

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
        processor = cached_get_processor(model_config.model)
        self.max_audio_clip_s = processor.feature_extractor.chunk_length
        self.model_sr = processor.feature_extractor.sampling_rate
        self.hop_length = processor.feature_extractor.hop_length

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params)


    async def _detect_language(
        self,
        audio_data: tuple[np.ndarray, int],
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> str:
        if request.language:
            if request.language in ISO639_1_SUPPORTED_LANGS:
                pass
            elif request.language in ISO639_1_OTHER_LANGS:
                logger.warning(
                    "The selected language %s has limited accuracy with"
                    " reported WER>=0.5. Results may be less accurate "
                    "for this choice.",
                    request.language,
                )
            else:
                raise ValueError(
                    f"Unsupported language: {request.language}."
                    "Language should be one of:"
                    + f" {list(ISO639_1_SUPPORTED_LANGS.values())}"
                    + f"or {list(ISO639_1_OTHER_LANGS.values())}"
                )
            return f"<|{request.language}|>"

        default_lang_token = "<|en|>"
        request_id = f"trsc-lang-{self._base_request_id(raw_request)}"

        # TODO: this mappping should be dynamice mapping once vllm core supports language dictionary
        if (
            "v3" in self.model_config.model.lower()
            and self.model_config.hf_config.model_type.lower() == "whisper"
        ):
            id2token = LANG_ID_TO_LANG_TOKEN["whisper-v3"]
        else:
            return default_lang_token

        prompt = cast(
            PromptType,
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": audio_data,
                    },
                },
                "decoder_prompt": "<|startoftranscript|>",
            },
        )
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=1,
            allowed_token_ids=list(id2token.keys()),
        )
        result_generator = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
        )

        lang_token = default_lang_token
        async for result in result_generator:
            lang_id = result.outputs[0].token_ids[0]
            lang_token = id2token[lang_id]
            break
        return lang_token


    async def _preprocess_transcription(
        self,
        request: TranscriptionRequest,
        audio_data: bytes,
        raw_request: Request,
        request_id: str,
    ) -> tuple[PromptType, float, str]:

        y, sr = load_audio(audio_data, request_id)

        if y is None:
            logger.info(f"{request_id} failed to load audio, y={y}, sr={sr}")
            raise ValueError(f"{request_id} Failed to load audio")

        assert isinstance(y, np.ndarray)
        assert isinstance(sr, int)

        duration = librosa.get_duration(y=y, sr=sr)

        if duration > self.max_audio_clip_s:
            raise ValueError(
                f"Maximum clip duration ({self.max_audio_clip_s}s) "
                "exceeded.")

        lang_token = await self._detect_language((y, sr), request, raw_request)

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (y, sr),
                },
            },
            "decoder_prompt":
            f"<|startoftranscript|>{lang_token}<|transcribe|>{request.prompt}"
        }
        return cast(PromptType, prompt), duration, lang_token

    # TODO (varun) : Make verbose response work !
    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest,
        raw_request: Request
    ) -> Union[TranscriptionResponse, TranscriptionResponseVerbose, AsyncGenerator[str, None],
               str, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.response_format not in ['text', 'json', 'verbose_json']:
            return self.create_error_response(
                "Currently only support response_format `text` or `json` or `verbose_json`.",)

        request_id = f"trsc-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            if lora_request:
                return self.create_error_response(
                    "Currently do not support LoRA for Transcription.")
            if prompt_adapter_request:
                return self.create_error_response(
                    "Currently do not support PromptAdapter for Transcription."
                )

            prompt, duration_s, lang_token = await self._preprocess_transcription(
                request=request,
                audio_data=audio_data,
                raw_request=raw_request,
                request_id=request_id,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        result_generator: Optional[AsyncGenerator[RequestOutput, None]] = None
        try:
            # TODO(rob): subtract len of tokenized prompt.
            default_max_tokens = self.model_config.max_model_len
            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params)
            sampling_params.logprobs = 1
            sampling_params.bad_words = ["<|notimestamps|>"]

            self._log_inputs(
                request_id,
                prompt['decoder_prompt'],  # type: ignore
                params=sampling_params,
                lora_request=None,
                prompt_adapter_request=None)

            result_generator = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if request.stream:
            return self.transcription_stream_generator(request,
                                                       result_generator,
                                                       request_id,
                                                       request_metadata,
                                                       duration_s)
        # Non-streaming response.
        try:
            assert result_generator is not None
            result = None
            async for op in result_generator:
                result = op
            assert isinstance(result, RequestOutput)
            tokenizer = await self.engine_client.get_tokenizer()
            completion_output = result.outputs[0]
            text = decode_tokens(tokenizer, list(completion_output.token_ids), skip_special_tokens=True).strip()
            if request.response_format == "json":
                return TranscriptionResponse(text=text)
            elif request.response_format == "text":
                return text
            elif request.response_format == "verbose_json":
                response = TranscriptionResponseVerbose(
                    task="transcribe",
                    duration=duration_s,
                    language=trim_lang_token(lang_token),
                    text=text,
                )
                segment_start, segment_end = None, None
                current_segment_text = ""
                current_segment_token_ids = []
                current_segment_logprobs = []
                logprobs_ptr = 0
                for token_id in completion_output.token_ids:
                    if token_id == 50257:
                        break
                    current_segment_token_ids.append(token_id)
                    current_segment_logprobs.append(completion_output.logprobs[logprobs_ptr][token_id].logprob)
                    logprobs_ptr += 1
                    if is_timestamp_token_id(token_id):
                        if segment_start is None:
                            segment_start = timestamp_token_id_to_timestamp(token_id)
                        else:
                            segment_end = timestamp_token_id_to_timestamp(token_id)
                            current_segment_text = decode_tokens(tokenizer, current_segment_token_ids,
                                                                 skip_special_tokens=True)
                            response.add_segment(
                                avg_logprob=float(np.mean(current_segment_logprobs)),
                                start=segment_start,
                                end=segment_end,
                                text=current_segment_text,
                                tokens=current_segment_token_ids,
                                temperature=request.temperature,
                            )
                            segment_start, segment_end, current_segment_text, current_segment_logprobs, current_segment_token_ids = None, None, "", [], []
                return response
            else:
                return self.create_error_response(
                    "Currently only support response_format `text`, `verbose_json` or `json`"
                )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def transcription_stream_generator(
            self, request: TranscriptionRequest,
            result_generator: AsyncGenerator[RequestOutput, None],
            request_id: str, request_metadata: RequestResponseMetadata,
            audio_duration_s: float) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model
        chunk_object_type: Final = "transcription.chunk"

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = request.stream_include_usage \
            if request.stream_include_usage else False
        include_continuous_usage = request.stream_continuous_usage_stats\
              if include_usage and request.stream_continuous_usage_stats\
                else False

        try:
            async for res in result_generator:
                # On first result.
                if res.prompt_token_ids is not None:
                    # Do not account the 4-tokens `<|startoftranscript|>..`
                    # Could be negative when language token is not specified.
                    num_prompt_tokens = max(len(res.prompt_token_ids) - 4, 0)
                    # NOTE(NickLucche) user can't pass encoder prompts directly
                    # at least not to Whisper. One indicator of the encoder
                    # amount of processing is the log-mel spectogram length.
                    num_prompt_tokens += ceil(audio_duration_s *
                                              self.model_sr / self.hop_length)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).

                # Just one output (n=1) supported.
                assert len(res.outputs) == 1
                output = res.outputs[0]

                delta_message = DeltaMessage(content=output.text)
                completion_tokens += len(output.token_ids)

                if output.finish_reason is None:
                    # Still generating, send delta update.
                    choice_data = TranscriptionResponseStreamChoice(
                        delta=delta_message)
                else:
                    # Model is finished generating.
                    choice_data = TranscriptionResponseStreamChoice(
                        delta=delta_message,
                        finish_reason=output.finish_reason,
                        stop_reason=output.stop_reason)

                chunk = TranscriptionStreamResponse(id=request_id,
                                                    object=chunk_object_type,
                                                    created=created_time,
                                                    choices=[choice_data],
                                                    model=model_name)

                # handle usage stats if requested & if continuous
                if include_continuous_usage:
                    chunk.usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=num_prompt_tokens + completion_tokens,
                    )

                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)

                final_usage_chunk = TranscriptionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens)

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
