from typing import Any, Dict, TypedDict, NotRequired

from collections.abc import Sequence
from vllm import PoolingRequestOutput, PoolingParams
from vllm.plugins.io_processors import IOProcessor
import asyncio


class EmbeddingPayload(TypedDict):
    model: str
    input: list[str] | str
    outputs: dict[str, bool]
    encoding_format: NotRequired[str]
    dimensions: NotRequired[int]
    user: NotRequired[str]
    normalize: NotRequired[bool]


class EmbeddingOutput(TypedDict):
    pass


class BGEM3IOProcessorPlugin(IOProcessor[list[str], Dict[str, Any]]):

    # No custom __init__: the base IOProcessor.__init__(vllm_config, renderer)
    # already stores vllm_config. Overriding it with a one-arg super().__init__
    # breaks on the v0.17.1 base, where `renderer` is a required positional.

    def parse_data(self, data: EmbeddingPayload) -> EmbeddingPayload:
        return data

    def pre_process(self, prompt: EmbeddingPayload, request_id: str | None = None, **kwargs) -> list[str]:
        if not (task := asyncio.current_task()):
            raise RuntimeError(f"BGEM3IOProcessorPlugin.pre_process must be called from within an async task.")

        task._bge_flags = {"dense": prompt["outputs"].get("dense"),
                           "colbert": prompt["outputs"].get("colbert"),
                           "sparse": prompt["outputs"].get("sparse"),
                           "normalize": prompt.get("normalize")}

        return prompt["input"] if isinstance(prompt["input"], list) else [prompt["input"]]

    def merge_pooling_params(self, params: PoolingParams | None = None) -> PoolingParams:
        out_flags = asyncio.current_task()._bge_flags
        return PoolingParams(task='embed', use_activation=bool(out_flags.get("normalize")), extra_kwargs={"outputs": out_flags})

    def post_process(self, model_output: Sequence[PoolingRequestOutput], request_id: str | None = None, **kwargs) -> \
    list[dict[str, Any]]:

        result: list[dict[str, Any]] = []

        for output in model_output:
            out_data = output.outputs.data[:3]
            seq_len = int(output.outputs.data[3].item())

            req_output: dict[str, Any] = {}
            offset = 4

            if out_data[0]:
                req_output["dense"] = output.outputs.data[offset: offset + 1024].tolist()
                offset += 1024
            if out_data[1]:
                colbert_sqz = output.outputs.data[offset: offset + (seq_len - 1) * 1024]
                colbert = colbert_sqz.reshape(seq_len - 1, 1024)
                req_output["colbert"] = colbert.tolist()
                offset += (seq_len - 1) * 1024

            if out_data[2]:
                unk_tok_id = 3
                sparse_weights = output.outputs.data[offset: offset + seq_len - 2].tolist()
                vocab_size = self.vllm_config.model_config.hf_config.vocab_size
                token_ids = output.prompt_token_ids[1:-1]
                sparse_vec = [0.0] * vocab_size
                for i, token_id in enumerate(token_ids):
                    w = sparse_weights[i]
                    if w > sparse_vec[token_id] and token_id != unk_tok_id:
                        sparse_vec[token_id] = w
                req_output["sparse"] = sparse_vec

            result.append(req_output)
        return result

