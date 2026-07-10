# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _compute_mhc_pre_num_split(
    *,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> int:
    block_k = 64
    block_m = 64
    k = hc_mult * hidden_size
    grid_size = cdiv(num_tokens, block_m)
    split_k = num_sms // grid_size
    num_block_k = cdiv(k, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _mhc_split_bucket_sizes(max_tokens: int, hc_hidden_size: int) -> list[int]:
    """One representative token size per distinct split-k value.

    ``num_tokens`` is a dynamic dim in the mhc_pre big-fuse kernels, but
    ``n_splits`` is a static compile key computed as
    ``num_sms // ceil(num_tokens / 64)`` (clamped). Runtime prefill sizes
    produce division values the power-of-two ladder misses, each triggering
    a multi-second JIT mid-serving. Enumerate every reachable ``n_splits``
    bucket and warm one size from each.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split

    block_m = 64
    sizes: list[int] = []
    seen: set[int] = set()
    for grid_size in range(1, cdiv(max_tokens, block_m) + 1):
        n_splits = compute_num_split(block_m, hc_hidden_size, grid_size)
        if n_splits not in seen:
            seen.add(n_splits)
            sizes.append(min(grid_size * block_m, max_tokens))
    return sizes


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _find_first_direct_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find a decoder layer that calls the mHC TileLang kernels directly.

    The NVIDIA DSv4 model (vllm/models/deepseek_v4/nvidia/model.py) does not
    wrap the kernels in ``hc_pre``/``hc_post`` ops; its forward calls
    ``mhc_pre_tilelang`` / ``mhc_fused_post_pre_tilelang`` free functions with
    the fused input-norm (``norm_weight=``).
    """
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "attn_norm",
                "ffn_norm",
                "hc_mult",
                "hidden_size",
                "rms_norm_eps",
                "hc_eps",
                "hc_post_alpha",
                "hc_sinkhorn_iters",
            )
        ):
            return module
    return None


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_direct_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    """Warm the kernel variants the NVIDIA DSv4 forward actually dispatches.

    Mirrors ``DeepseekV4DecoderLayer.forward``: ``mhc_pre_tilelang`` with the
    fused attn-norm (compiles ``mhc_pre_big_fuse_with_norm_tilelang`` per
    split-k bucket), the post+pre fused op with the fused ffn-norm, and the
    standalone ``mhc_post_tilelang`` used at the end of the stack / for aux
    hidden states. ``num_tokens`` is a dynamic dim in these kernels, so
    repeated sizes mapping to the same static config hit TileLang's cache.
    """
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    x = torch.zeros(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    for size in token_sizes:
        residual_slice = residual[:size]
        post_mix, res_mix, _ = mhc_pre_tilelang(
            residual_slice,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            norm_weight=layer.attn_norm.weight.data,
            norm_eps=layer.attn_norm.variance_epsilon,
        )
        residual_cur, post_mix, res_mix, _ = mhc_fused_post_pre_tilelang(
            x[:size],
            residual_slice,
            post_mix,
            res_mix,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=layer.ffn_norm.weight.data,
            norm_eps=layer.ffn_norm.variance_epsilon,
        )
        mhc_post_tilelang(x[:size], residual_cur, post_mix, res_mix)


def _warmup_direct_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang,
    )

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_fused_kernel_tilelang(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    direct_layer = None if layer is not None else _find_first_direct_mhc_layer(model)
    if layer is None and direct_layer is None:
        logger.warning(
            "DeepSeek V4 model has no recognizable mHC layer structure; "
            "skipping mHC TileLang warmup (kernels will JIT on first use)."
        )
        return

    device = (layer or direct_layer).hc_attn_fn.device
    if device.type != "cuda":
        return

    deepseek_model = _find_deepseek_v4_model(model)
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if token_sizes and direct_layer is not None:
        # The direct path dispatches split-k big-fuse kernels; cover every
        # reachable n_splits bucket, not just the power-of-two ladder.
        hc_hidden_size = int(direct_layer.hc_mult) * int(direct_layer.hidden_size)
        token_sizes = _normalize_token_sizes(
            token_sizes + _mhc_split_bucket_sizes(max(token_sizes), hc_hidden_size),
            max_tokens=max(token_sizes),
        )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        if layer is not None:
            _warmup_layer_mhc(layer, token_sizes)
        else:
            _warmup_direct_layer_mhc(direct_layer, token_sizes)
        if deepseek_model is not None:
            if getattr(deepseek_model, "hc_head_op", None) is not None:
                _warmup_hc_head(deepseek_model, token_sizes)
            elif all(
                hasattr(deepseek_model, attr) for attr in ("rms_norm_eps", "hc_eps")
            ):
                _warmup_direct_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
