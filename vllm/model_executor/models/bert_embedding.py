from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import BertConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import LoRAConfig
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import EmbeddingSequenceGroupOutput, SamplerOutput

class BertSelfOutput(nn.Module):
    def __init__(
		self,
        hidden_size: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        linear_method: Optional[LinearMethodBase] = None,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.head_dim = hidden_size // self.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.num_attention_heads,
            self.num_attention_heads,
            bias,
            linear_method=linear_method,
        )
        self.attn = Attention(self.num_attention_heads,
                              self.head_dim,
                              self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # TODO(Iskren): attention_mask -- probably somehow encoded in attn_metadata
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([
            self.num_attention_heads,
            self.num_attention_heads,
            self.num_attention_heads], dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        return attn_output


class BertAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        linear_method: Optional[LinearMethodBase] = None,
        bias: bool = True
    ):
        super().__init__()
        self.self = BertSelfAttention(
            hidden_size,
            num_attention_heads,
            linear_method,
            bias)
        self.output = BertSelfOutput(hidden_size, layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor]:
        self_output = self.self(
            hidden_states,
            kv_cache,
            attn_metadata,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output

def _get_bert_act_fn(hidden_act: Optional[str]):
    if hidden_act is None or hidden_act == "gelu":
        return GeluAndMul(approximate="none")
    raise ValueError(f"unsupported act fn {hidden_act}")

class BertIntermediate(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: Optional[str],
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = _get_bert_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        linear_method: Optional[LinearMethodBase] = None,
        # hidden_size: int,
        # num_attention_heads: int,
        # layer_norm_eps: float,
        # bias: bool,
        # intermediate_size: int,
        # hidden_act: Optional[str],
    ):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = BertAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.layer_norm_eps,
            linear_method,
            bias=True, # where to get this?
        )
        self.intermediate = BertIntermediate(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act)
        self.output = BertOutput(
            config.hidden_size,
            config.intermediate_size,
            config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor]:
        attention_output = self.attention(
            hidden_states,
            kv_cache,
            attn_metadata,
        )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        hidden_size: int,
        layer_norm_eps: float,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size + lora_vocab,
            hidden_size,
            org_num_embeddings=vocab_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        embeds = self.word_embeddings(input_ids)
        embeds += self.position_embeddings(position_ids)
        if token_type_ids:
            embeds += self.token_type_embeddings(token_type_ids)
        embeds = self.LayerNorm(embeds)

        return embeds

class BertEncoder(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata
    ):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )

class BertEmbeddingModel(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(
            config.vocab_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            config.hidden_size,
            config.layer_norm_eps,
            lora_config=lora_config,
        )
        self.encoder = BertEncoder(config, linear_method)
        # self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # pooler? -- config is in 1_Pooling/config.json,
        # - for e5-base-v2, gte-large it's mean
        # - for bge-large-en-v1.5 it's cls (first)
        # - for e5-mistral it's last (but no 1_Pooling/config.json)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids, positions)

        hidden_states = self.encoder(input_ids, positions, kv_caches, attn_metadata)


        # hidden_states = self.norm(hidden_states, ...)
        return hidden_states

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            # if "rotary_emb.inv_freq" in name:
            #     continue
            # if ("rotary_emb.cos_cached" in name
            #         or "rotary_emb.sin_cached" in name):
            #     # Models trained using ColossalAI may include these tensors in
            #     # the checkpoint. Skip them.
            #     continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
