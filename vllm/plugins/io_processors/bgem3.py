from typing import Any, Dict, TypedDict, NotRequired

from collections.abc import Sequence
from vllm import PoolingRequestOutput, PoolingParams
from vllm.config import VllmConfig
from vllm.plugins.io_processors import IOProcessor
import asyncio


class EmbeddingPayload(TypedDict):
    model: str
    input: list[str] | str
    outputs: dict[str, bool]
    encoding_format: NotRequired[str]
    dimensions: NotRequired[int]
    user: NotRequired[str]


class EmbeddingOutput(TypedDict):
    pass


class BGEM3IOProcessorPlugin(IOProcessor[list[str], Dict[str, Any]]):

    def __init__(self, config: VllmConfig):
        super().__init__(config)

    def parse_data(self, data: EmbeddingPayload) -> EmbeddingPayload:
        return data

    def pre_process(self, prompt: EmbeddingPayload, request_id: str | None = None, **kwargs) -> list[str]:
        if not (task := asyncio.current_task()):
            raise RuntimeError(f"BGEM3IOProcessorPlugin.pre_process must be called from within an async task.")

        task._bge_flags = {"dense": prompt["outputs"].get("dense"),
                           "colbert": prompt["outputs"].get("colbert"),
                           "sparse": prompt["outputs"].get("sparse")}

        return prompt["input"] if isinstance(prompt["input"], list) else [prompt["input"]]

    def merge_pooling_params(self, params: PoolingParams | None = None) -> PoolingParams:
        out_flags = asyncio.current_task()._bge_flags
        return PoolingParams(task='embed', extra_kwargs={"outputs": out_flags})

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
                sparse_weights = output.outputs.data[offset: offset + seq_len - 2].tolist()
                sparse = {}
                for i, token_id in enumerate(output.prompt_token_ids[1: -1]):
                    if token_id not in sparse:
                        sparse[token_id] = sparse_weights[i]
                    else:
                        sparse[token_id] = max(sparse[token_id], sparse_weights[i])
                req_output["sparse"] = sparse

            result.append(req_output)

        return result

