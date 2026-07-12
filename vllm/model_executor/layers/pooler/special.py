# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Set
from itertools import groupby

import torch

from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .abstract import Pooler, PoolerOutput
from .common import ClassifierFn
from .seqwise import (
    SequencePoolingFn,
    SequencePoolingMethod,
    pooler_for_classify,
    pooler_for_embed,
)
from .tokwise import AllPool, pooler_for_token_classify, pooler_for_token_embed


class DispatchPooler(Pooler):
    """Dispatches calls to a sub-pooler based on the pooling task."""

    @classmethod
    def for_embedding(cls, pooler_config: PoolerConfig):
        return cls(
            {
                "token_embed": pooler_for_token_embed(pooler_config),
                "embed": pooler_for_embed(pooler_config),
            },
        )

    @classmethod
    def for_seq_cls(
        cls,
        pooler_config: PoolerConfig,
        *,
        pooling: SequencePoolingMethod | SequencePoolingFn | None = None,
        classifier: ClassifierFn | None = None,
    ):
        return cls(
            {
                "token_classify": pooler_for_token_classify(
                    pooler_config,
                    pooling=AllPool(),
                    classifier=classifier,
                ),
                "classify": pooler_for_classify(
                    pooler_config,
                    pooling=pooling,
                    classifier=classifier,
                    act_fn="classify",
                ),
                "score": pooler_for_classify(
                    pooler_config,
                    pooling=pooling,
                    classifier=classifier,
                    act_fn="score",
                ),
            }
        )

    def __init__(self, poolers_by_task: Mapping[PoolingTask, Pooler]) -> None:
        super().__init__()

        for task, pooler in poolers_by_task.items():
            if task not in pooler.get_supported_tasks():
                raise ValueError(
                    f"{pooler=} does not support {task=}. "
                    f"Supported tasks: {pooler.get_supported_tasks()}"
                )

        self.poolers_by_task = poolers_by_task

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return set(self.poolers_by_task)

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)



    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:



        outputs: list[torch.Tensor] = []
        batch_size = len(pooling_metadata.prompt_lens)

        for p in pooling_metadata.pooling_params:
            p.use_activation = True #always normalize dense and colbert to match the processor_flag_embeddings implementation
        dense_pooler = self.poolers_by_task['embed']
        dense = dense_pooler(hidden_states, pooling_metadata)
        colbert_pooler = self.poolers_by_task['token_embed']
        colbert = colbert_pooler(hidden_states, pooling_metadata)
        sparse_pooler = self.poolers_by_task['token_classify']
        sparse = sparse_pooler(hidden_states, pooling_metadata)


        for i in range(batch_size):

            # Warmup (_dummy_pooler_run) calls the pooler with dummy metadata that has
            # no "outputs" flag. Default to emitting all three heads so the dummy run
            # both survives and exercises the largest output (correct memory profiling).
            extra = pooling_metadata.pooling_params[i].extra_kwargs or {}
            out_mask = extra.get("outputs") or {"dense": True, "colbert": False, "sparse": False}
            out_data = torch.zeros([3], dtype=torch.float32, device= hidden_states.device)
            dense_req = torch.empty(0, dtype=torch.float32, device= hidden_states.device)
            colbert_req = torch.empty(0, dtype=torch.float32, device= hidden_states.device)
            sparse_req = torch.empty(0, dtype=torch.float32, device= hidden_states.device)
            if out_mask["dense"]:
                dense_req = dense[i].flatten().to(torch.float32)
                out_data[0] = 1
            if out_mask["colbert"]:
                colbert_req = colbert[i].flatten().to(torch.float32)
                out_data[1] = 1
            if out_mask["sparse"]:
                sparse_req = sparse[i].flatten().to(torch.float32)
                out_data[2] = 1

            seq_len = pooling_metadata.prompt_lens[i].item()
            metadata_len = torch.tensor([seq_len], dtype= torch.float32, device= hidden_states.device)
            output_req = torch.cat([out_data, metadata_len, dense_req, colbert_req, sparse_req], dim=0)
            outputs.append(output_req)

        return outputs

    def extra_repr(self) -> str:
        s = f"supported_task={self.get_supported_tasks()}"
        return s


class IdentityPooler(Pooler):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"plugin", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        return hidden_states


class BOSEOSFilter(Pooler):
    """Filters the BOS and EOS token results from outputs."""

    def __init__(
        self,
        pooler: Pooler,
        bos_token_id: int = -1,  # -1 disables the filtering
        eos_token_id: int = -1,
    ) -> None:
        super().__init__()

        self.pooler = pooler
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooler.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_outputs = self.pooler(hidden_states, pooling_metadata)
        assert isinstance(pooled_outputs, list)

        for i, prompt_len in enumerate(pooling_metadata.prompt_lens):
            pooled_data = pooled_outputs[i]
            assert (
                isinstance(pooled_data, torch.Tensor)
                and pooled_data.shape[0] == prompt_len
            )
            token_ids = pooling_metadata.prompt_token_ids[i, :prompt_len]
            if token_ids[0] == self.bos_token_id:
                pooled_data = pooled_data[1:]
            if token_ids[-1] == self.eos_token_id:
                pooled_data = pooled_data[:-1]
            pooled_outputs[i] = pooled_data.squeeze(-1)

        return pooled_outputs


__all__ = ["BOSEOSFilter", "DispatchPooler", "IdentityPooler"]
