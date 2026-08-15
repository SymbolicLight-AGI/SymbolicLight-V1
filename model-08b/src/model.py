#!/usr/bin/env python3
"""SymbolicLight V1 model implementation."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SymbolicLightConfig:
    """Default configuration for SymbolicLight V1."""
    vocab_size: int = 57344
    embed_dim: int = 1536
    n_layers: int = 22
    n_heads: int = 24
    head_dim: int = 64
    intermediate_dim: int = 6144
    max_seq_len: int = 512
    spike_chunk_size: int = 64
    dropout: float = 0.1

    spike_threshold: float = 1.0
    leak_factor: float = 0.95
    rope_theta: float = 10000.0
    frontend_mode: str = "text"

    sparse_attn_window: int = 512
    n_global_anchors: int = 4
    enable_sparse_attn: bool = True
    enable_dynamic_prior: bool = True

def hard_spike(membrane_potential: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Apply the inference-time hard spike threshold."""
    return (membrane_potential >= threshold).float()

class RotaryPositionEncoding(nn.Module):
    """Rotary position embedding."""
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE rotation to the input tensor.
        Args:
            x: [B, S, D] input continuous representation
            offset: position offset for incremental decoding
        Returns:
            rotated: [B, S, D] rotated representation
        """
        B, S, D = x.shape

        t = torch.arange(offset, offset + S, device=x.device, dtype=torch.float32)

        freqs = torch.outer(t, self.inv_freq.to(x.device))

        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos().unsqueeze(0)
        sin_emb = emb.sin().unsqueeze(0)

        x_rotated = torch.cat([
            -x[..., D // 2:],
             x[..., :D // 2],
        ], dim=-1)

        return x * cos_emb + x_rotated * sin_emb

class TextEmbeddingFrontend(nn.Module):
    """Text-token embedding frontend for this artifact release."""
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

    def forward(self, token_ids: torch.Tensor, modality: str = "text") -> torch.Tensor:
        if modality != "text":
            raise NotImplementedError(
                "This artifact release exposes the text-token path only."
            )
        return self.text_embedding(token_ids)

def _lif_scan_forward(x: torch.Tensor, v_mem: torch.Tensor,
                     leak: float, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT forward pass for temporal LIF neuron scanning.
    Returns: (spikes [B,S,D], final_v_mem [B,D])
    """
    B, S, D = x.shape
    spikes = torch.empty_like(x)
    for t in range(S):
        v_mem = v_mem * leak + x[:, t, :]
        v_mem = torch.clamp(v_mem, -3.0, 3.0)
        spike = (v_mem >= threshold).float()
        v_mem = v_mem * (1.0 - spike)
        spikes[:, t, :] = spike
    return spikes, v_mem

class SpikeEncoder(nn.Module):
    """
    Convert discrete token IDs into spatiotemporal spike tensors.

    Main design updates:
    - remove learned positional embeddings and use RoPE in SparseTCAM
    - use chunk-parallel LIF spike conversion to reduce Python loops
    - route token embeddings through a text-only frontend instead of a hard-coded embedding

    Flow: token_id -> TextEmbeddingFrontend -> LayerNorm -> parallel LIF spike conversion
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config

        self.frontend = TextEmbeddingFrontend(config)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.threshold = config.spike_threshold
        self.leak = config.leak_factor

        self.v_mem = None

    def _init_membrane(self, shape: torch.Size, device: torch.device):
        """Initialize or reset the membrane potential."""
        self.v_mem = torch.zeros(shape, device=device)

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False,
                cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len]
            use_cache: if True, run O(1) incremental decoding
            cache: cache dictionary
        Returns:
            spikes: [batch, seq_len, embed_dim] sparse 0/1 spikes
            continuous: [batch, seq_len, embed_dim] continuous residual stream
        """
        B, S = token_ids.shape

        if use_cache and cache is not None:
            if 'v_mem' not in cache:
                cache['v_mem'] = torch.zeros(B, self.config.embed_dim, device=token_ids.device)
            if 'seq_len' not in cache:
                cache['seq_len'] = 0
            self.v_mem = cache['v_mem']
            cache['seq_len'] += S
        else:
            self._init_membrane((B, self.config.embed_dim), token_ids.device)

        x = self.frontend(token_ids)
        x = self.norm(x)

        chunk_size = self.config.spike_chunk_size
        spikes_list = []

        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            x_chunk = x[:, chunk_start:chunk_end, :]
            chunk_spikes, self.v_mem = _lif_scan_forward(
                x_chunk, self.v_mem, self.leak, self.threshold
            )
            spikes_list.append(chunk_spikes)

        spikes = torch.cat(spikes_list, dim=1)

        if use_cache and cache is not None:
            cache['v_mem'] = self.v_mem.detach()

        return spikes, x

class SparseLocalAttention(nn.Module):
    """
    Compute attention only among active spike positions with a local window and global anchors.

    Key idea:
    - dense attention attends across all S positions -> O(S^2)
    - this path only attends over active positions inside a local window -> O(S * k * w)
      where k is the active fraction and w is the window size
    - global anchors let the first few tokens interact broadly and stabilize global context

    Relation to the decay path:
    - the decay path compresses history into a fixed-size hidden state for coarse long-range memory
    - the attention path focuses precisely on recent informative positions for local reasoning
    - a learned gate blends both paths
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.window_size = max(1, int(config.sparse_attn_window))
        self.n_global_anchors = config.n_global_anchors
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        self._use_sdpa = hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self.rope = RotaryPositionEncoding(config.head_dim, theta=config.rope_theta)

        S = config.max_seq_len
        q_pos = torch.arange(S).unsqueeze(1)
        k_pos = torch.arange(S).unsqueeze(0)
        distance = q_pos - k_pos
        causal = distance >= 0
        window = (q_pos - k_pos) <= self.window_size
        anchors = k_pos < self.n_global_anchors
        self.register_buffer('_cached_mask', causal & (window | anchors))

    def _trim_kv_cache(self, K: torch.Tensor, V: torch.Tensor,
                       spike_mask: torch.Tensor,
                       positions: torch.Tensor):
        """Keep global anchor tokens plus the recent sparse attention window."""
        if positions.numel() == 0:
            return K, V, spike_mask, positions

        recent_start = positions[-1] - self.window_size
        keep = (positions < self.n_global_anchors) | (positions >= recent_start)
        if bool(keep.all()):
            return K, V, spike_mask, positions

        return (
            K[:, :, keep, :],
            V[:, :, keep, :],
            spike_mask[:, keep],
            positions[keep],
        )

    def forward(self, x: torch.Tensor, spike_mask: torch.Tensor,
                offset: int = 0, use_cache: bool = False, cache: dict = None) -> torch.Tensor:
        """
        Args:
            x: [B, S_q, D] continuous representation; RoPE is applied internally to Q/K
            spike_mask: [B, S_q] boolean mask, True means the position fired a spike
            offset: RoPE position offset for incremental decoding
            use_cache: whether to use the KV cache for incremental decoding
            cache: KV cache dictionary
        Returns:
            attn_out: [B, S_q, D] sparse attention output with zeros on inactive positions
        """
        B, S_q, D = x.shape

        Q = self.q_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)

        Q = self.rope(Q.contiguous().view(B * self.n_heads, S_q, self.head_dim), offset=offset)
        Q = Q.view(B, self.n_heads, S_q, self.head_dim).to(V.dtype)
        K = self.rope(K.contiguous().view(B * self.n_heads, S_q, self.head_dim), offset=offset)
        K = K.view(B, self.n_heads, S_q, self.head_dim).to(V.dtype)

        positions = None
        if use_cache and cache is not None:
            current_positions = torch.arange(
                offset, offset + S_q, device=x.device, dtype=torch.long
            )
            if 'K' in cache:
                cached_positions = cache.get('positions')
                if cached_positions is None:
                    cached_positions = torch.arange(
                        0, cache['K'].size(2), device=x.device, dtype=torch.long
                    )
                K = torch.cat([cache['K'], K], dim=2)
                V = torch.cat([cache['V'], V], dim=2)
                spike_mask_kv = torch.cat([cache['spike_mask'], spike_mask], dim=1)
                positions = torch.cat([cached_positions, current_positions], dim=0)
            else:
                spike_mask_kv = spike_mask
                positions = current_positions

            if S_q == 1:
                K, V, spike_mask_kv, positions = self._trim_kv_cache(
                    K, V, spike_mask_kv, positions
                )
            cache['K'] = K.detach()
            cache['V'] = V.detach()
            cache['spike_mask'] = spike_mask_kv.detach()
            cache['positions'] = positions.detach()
        else:
            spike_mask_kv = spike_mask
            positions = torch.arange(0, K.size(2), device=x.device, dtype=torch.long)

        S_kv = K.size(2)

        if offset == 0 and S_q == S_kv and S_q == self._cached_mask.size(0):
            attn_mask = self._cached_mask
        else:
            q_pos = torch.arange(offset, offset + S_q, device=x.device).unsqueeze(1)
            k_pos = positions.unsqueeze(0)
            distance = q_pos - k_pos
            causal = distance >= 0
            window = distance <= self.window_size
            anchors = k_pos < self.n_global_anchors
            attn_mask = causal & (window | anchors)

        spike_key_mask = spike_mask_kv.unsqueeze(1).unsqueeze(2)
        full_mask = attn_mask.unsqueeze(0).unsqueeze(0) & spike_key_mask

        query_has_any_key = full_mask.any(dim=-1, keepdim=True)
        if self._use_sdpa:
            safe_mask = full_mask | ~query_has_any_key
            attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=safe_mask, dropout_p=0.0)
            attn_out = attn_out.masked_fill(~query_has_any_key, 0.0)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~full_mask, float('-inf'))
            scores = scores.masked_fill(~query_has_any_key, 0.0)
            attn_weights = F.softmax(scores, dim=-1).to(V.dtype)
            attn_weights = attn_weights.masked_fill(~query_has_any_key, 0.0)
            attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, D)

        attn_out = attn_out * spike_mask.unsqueeze(-1).to(dtype=attn_out.dtype)

        if use_cache and cache is not None and S_q > 1:
            cache['K'], cache['V'], cache['spike_mask'], cache['positions'] = [
                tensor.detach() for tensor in self._trim_kv_cache(
                    cache['K'], cache['V'], cache['spike_mask'], cache['positions']
                )
            ]

        return attn_out

class SparseTCAM(nn.Module):
    """Dual-path spike-gated sequence mixer."""
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.threshold = config.spike_threshold
        self.leak = config.leak_factor
        self.enable_sparse_attn = config.enable_sparse_attn

        self.tcam_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.decay_raw = nn.Parameter(torch.full((config.n_heads,), 3.0))

        if self.enable_sparse_attn:
            self.sparse_attn = SparseLocalAttention(config)

        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, spikes: torch.Tensor, continuous: torch.Tensor,
                use_cache: bool = False, cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main updates:
        1. apply RoPE inside SparseLocalAttention after Q/K projection
        2. support cached hidden-state reads and writes across decoding chunks
        """
        B, S, D = spikes.shape
        compute_dtype = continuous.dtype
        if spikes.dtype != compute_dtype:
            spikes = spikes.to(compute_dtype)

        offset = 0
        if use_cache and cache is not None:
            offset = cache.get('rope_offset', 0)
            cache['rope_offset'] = offset + S

        spike_energy = spikes.sum(dim=-1)
        active_mask = (spike_energy > 0).unsqueeze(-1).to(dtype=compute_dtype)

        tcam_out = self.tcam_proj(spikes * active_mask)

        tcam_out = tcam_out.view(B, S, self.n_heads, self.head_dim)

        decay = torch.sigmoid(self.decay_raw)

        if cache is not None:
            if 'h' not in cache:
                cache['h'] = torch.zeros(B, self.n_heads, self.head_dim, device=spikes.device, dtype=compute_dtype)
            h = cache['h']
        else:
            h = torch.zeros(B, self.n_heads, self.head_dim, device=spikes.device, dtype=compute_dtype)

        if use_cache and cache is not None and S == 1:

            h = decay.view(1, self.n_heads, 1) * h + (1 - decay.view(1, self.n_heads, 1)) * tcam_out[:, 0]
            cache['h'] = h.detach()
            context = h.unsqueeze(1)
        else:

            powers = torch.arange(S - 1, -1, -1, dtype=compute_dtype, device=spikes.device)
            kernel = ((decay.view(-1, 1) ** powers.view(1, -1)) * (1 - decay).view(-1, 1)).unsqueeze(1)
            tcam_out_trans = tcam_out.permute(0, 3, 2, 1).reshape(-1, self.n_heads, S)
            tcam_out_pad = F.pad(tcam_out_trans, (S - 1, 0))
            out = F.conv1d(tcam_out_pad, kernel, groups=self.n_heads)
            context = out.view(-1, self.head_dim, self.n_heads, S).permute(0, 3, 2, 1)

            powers_fwd = torch.arange(1, S + 1, dtype=compute_dtype, device=spikes.device).view(1, S, 1, 1)
            decay_t = decay.view(1, 1, self.n_heads, 1) ** powers_fwd
            context = context + h.unsqueeze(1) * decay_t

            if cache is not None:
                cache['h'] = context[:, -1, :, :].detach()

        decay_output = context.reshape(B, S, D)

        if self.enable_sparse_attn:
            spike_mask = (spikes.sum(dim=-1) > 0)

            attn_cache = cache.setdefault('attn', {}) if cache is not None else None
            attn_output = self.sparse_attn(
                continuous, spike_mask, offset=offset,
                use_cache=use_cache, cache=attn_cache
            )

            gate = torch.sigmoid(self.attn_gate)
            output = gate * attn_output + (1 - gate) * decay_output
        else:
            output = decay_output

        output = self.out_proj(self.dropout(output))

        out_continuous = self.norm(continuous + output)

        out_spikes = hard_spike(out_continuous, self.threshold).to(out_continuous.dtype)

        return out_spikes, out_continuous

class SpikingFeedForward(nn.Module):
    """
    Two-layer feed-forward block used in place of the standard Transformer MLP.
    The main difference is the LIF-style spike activation in the hidden layer.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.up = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.down = nn.Linear(config.intermediate_dim, config.embed_dim, bias=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.threshold = config.spike_threshold
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.up(x)
        h = hard_spike(h, self.threshold).to(x.dtype)
        h = self.down(self.dropout(h))
        return self.norm(residual + h)

class SymbolicLightBlock(nn.Module):
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.tcam = SparseTCAM(config)
        self.ffn = SpikingFeedForward(config)

    def forward(self, spikes, continuous, use_cache=False, cache=None):
        spikes, continuous = self.tcam(spikes, continuous, use_cache=use_cache, cache=cache)
        continuous = self.ffn(continuous)
        spikes = hard_spike(continuous, self.tcam.threshold).to(continuous.dtype)
        return spikes, continuous

class BayesianHead(nn.Module):
    """
    Dynamic context-conditioned prior head.

    Earlier versions used a static learned log_prior vector.
    This version predicts log_prior from the current context with a lightweight network.

    Bayesian form:
      log P(word|context) = log P(context|word) + log P(word|context_summary)
                           likelihood term            dynamic prior term

    Intuition:
    - when the context is about cooking, the prior can upweight tokens such as salt or pan
    - when the context is about programming, the prior can upweight tokens such as function or loop
    - this is more targeted than a static frequency bias
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.use_dynamic_prior = getattr(config, 'enable_dynamic_prior', True)

        self.prior_weight = nn.Parameter(torch.tensor(0.1))

        if self.use_dynamic_prior:
            bottleneck_dim = config.embed_dim // 4
            self.prior_net = nn.Sequential(
                nn.Linear(config.embed_dim, bottleneck_dim, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck_dim, config.vocab_size, bias=False),
            )
        else:
            self.log_prior = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, continuous: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous: [B, S, D]
        Returns:
            logits: [B, S, vocab_size]
        """
        log_likelihood = self.output_proj(continuous)

        if self.use_dynamic_prior:
            dynamic_prior = self.prior_net(continuous)
            logits = log_likelihood + self.prior_weight * dynamic_prior
        else:
            logits = log_likelihood + self.prior_weight * self.log_prior

        return logits

class SymbolicLightModel(nn.Module):
    """SymbolicLight language model."""
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        self.spike_encoder = SpikeEncoder(config)
        self.blocks = nn.ModuleList([
            SymbolicLightBlock(config) for _ in range(config.n_layers)
        ])
        self.output_head = BayesianHead(config)
        self.apply(self._init_weights)

        self.parameter_count = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compile_for_inference(self):
        """Apply torch.compile to selected inference-critical submodules."""
        self.spike_encoder = torch.compile(self.spike_encoder, mode='reduce-overhead')
        for block in self.blocks:
            block.tcam = torch.compile(block.tcam, mode='reduce-overhead')
            block.ffn = torch.compile(block.ffn, mode='reduce-overhead')
        return self

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False,
                past_key_values: list = None,
                last_logits_only: bool = False):
        """
        Inference-only forward pass.

        Args:
            token_ids: [B, S] input token IDs
            use_cache: whether to use the KV cache for inference
            past_key_values: list of inference caches
            last_logits_only: if True, project only the final sequence position
        Returns:
            logits: [B, S, vocab_size] by default, or [B, 1, vocab_size]
                    when last_logits_only=True
        """

        if use_cache and past_key_values is None:
            past_key_values = [{} for _ in range(len(self.blocks) + 1)]

        caches = past_key_values if use_cache else [None] * (len(self.blocks) + 1)

        encoder_cache = caches[0] if caches[0] is not None else (
            past_key_values[0] if use_cache else None
        )
        spikes, continuous = self.spike_encoder(token_ids, use_cache=use_cache, cache=encoder_cache)
        model_dtype = self.output_head.output_proj.weight.dtype
        if continuous.dtype != model_dtype:
            continuous = continuous.to(model_dtype)
        if spikes.dtype != model_dtype:
            spikes = spikes.to(model_dtype)
        for i, block in enumerate(self.blocks):
            block_cache = caches[i + 1] if caches[i + 1] is not None else (
                past_key_values[i + 1] if use_cache else None
            )
            spikes, continuous = block(
                spikes, continuous,
                use_cache=use_cache, cache=block_cache,
            )

        output_continuous = continuous[:, -1:, :] if last_logits_only else continuous
        logits = self.output_head(output_continuous)

        return logits

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 1.0, repetition_penalty: float = 1.0,
                 adaptive_temperature: bool = True,
                 eos_token_id: Optional[int] = None,
                 stop_token_id: Optional[int] = None,
                 tokenizer=None,
                 valid_vocab_size: Optional[int] = None,
                 banned_token_ids: Optional[List[int]] = None,
                 diagnostics: Optional[dict] = None) -> torch.Tensor:
        """
        Autoregressive text generation with bounded-window cached decoding.

        Adaptive temperature:
          - lower entropy -> lower temperature for more deterministic outputs
          - higher entropy -> higher temperature for more exploratory outputs
          - effective range is approximately [0.5 x base_temp, 1.5 x base_temp]
        """
        if max_new_tokens <= 0:
            return prompt_ids.clone()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if not (0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        self.eval()
        past_key_values = [{} for _ in range(len(self.blocks) + 1)]
        batch_size = prompt_ids.size(0)
        recent_ids: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)
        generated_tokens: list[torch.Tensor] = []
        seen_token_ids = [set(row.tolist()) for row in prompt_ids]
        assistant_end = "<|endofassistant|>"
        invalid_masses: list[float] = []
        banned_masses: list[float] = []

        def _resolve_eos_token_id():
            if eos_token_id is not None:
                return eos_token_id
            if tokenizer is not None:
                eos_id_value = getattr(tokenizer, "eos_id", None)
                if eos_id_value is not None:
                    return eos_id_value() if callable(eos_id_value) else eos_id_value
            return 2

        def _resolve_stop_token_id():
            if stop_token_id is not None:
                return stop_token_id
            if tokenizer is None:
                return None
            piece_to_id = getattr(tokenizer, "piece_to_id", None)
            if piece_to_id is not None:
                token_id = piece_to_id(assistant_end)
                if token_id is None or int(token_id) < 0:
                    return None
                id_to_piece = getattr(tokenizer, "id_to_piece", None)
                if id_to_piece is not None and id_to_piece(int(token_id)) != assistant_end:
                    return None
                return int(token_id)
            return None

        def _resolve_valid_vocab_size():
            if valid_vocab_size is not None:
                return int(valid_vocab_size)
            if tokenizer is not None:
                vocab_size_value = getattr(tokenizer, "vocab_size", None)
                if vocab_size_value is not None:
                    return int(vocab_size_value() if callable(vocab_size_value) else vocab_size_value)
            return None

        resolved_eos_token_id = int(_resolve_eos_token_id())
        assistant_stop_token_id = _resolve_stop_token_id()
        resolved_valid_vocab_size = _resolve_valid_vocab_size()
        if resolved_valid_vocab_size is not None and resolved_valid_vocab_size <= 0:
            raise ValueError("valid_vocab_size must be > 0")

        resolved_banned_token_ids = []
        if banned_token_ids:
            vocab_size = int(self.config.vocab_size)
            allowed_stop_ids = {resolved_eos_token_id}
            if assistant_stop_token_id is not None:
                allowed_stop_ids.add(int(assistant_stop_token_id))
            seen_banned = set()
            for token_id in banned_token_ids:
                token_id = int(token_id)
                if token_id in allowed_stop_ids:
                    continue
                if token_id < 0 or token_id >= vocab_size or token_id in seen_banned:
                    continue
                seen_banned.add(token_id)
                resolved_banned_token_ids.append(token_id)
        banned_token_tensor = (
            torch.tensor(resolved_banned_token_ids, device=prompt_ids.device, dtype=torch.long)
            if resolved_banned_token_ids
            else None
        )

        logits = self.forward(
            prompt_ids,
            use_cache=True,
            past_key_values=past_key_values,
            last_logits_only=True,
        )

        def _adaptive_temp(raw_logits, base_temp):
            if not adaptive_temperature:
                return base_temp
            probs = F.softmax(raw_logits.float(), dim=-1)
            p = probs.clamp(1e-7, 1.0)
            entropy = -(p * p.log()).sum(dim=-1).mean()
            entropy_vocab_size = self.config.vocab_size
            if resolved_valid_vocab_size is not None:
                entropy_vocab_size = min(entropy_vocab_size, resolved_valid_vocab_size)
            max_entropy = math.log(max(entropy_vocab_size, 2))
            norm_entropy = (entropy / max_entropy).clamp(0, 1).item()
            min_temp = max(0.1, base_temp * 0.5)
            max_temp = min(1.5, base_temp * 1.5)
            max_temp = max(max_temp, min_temp)
            return min_temp + norm_entropy * (max_temp - min_temp)

        def _record_invalid_mass(raw_logits):
            if diagnostics is None or resolved_valid_vocab_size is None:
                return
            vocab_size = raw_logits.size(-1)
            if resolved_valid_vocab_size >= vocab_size:
                invalid_masses.append(0.0)
                return
            probs = F.softmax(raw_logits.float(), dim=-1)
            invalid_mass = probs[:, resolved_valid_vocab_size:].sum(dim=-1).mean()
            invalid_masses.append(float(invalid_mass.detach().cpu().item()))

        def _record_banned_mass(raw_logits):
            if diagnostics is None or banned_token_tensor is None:
                return
            probs = F.softmax(raw_logits.float(), dim=-1)
            banned_mass = probs.index_select(dim=-1, index=banned_token_tensor).sum(dim=-1).mean()
            banned_masses.append(float(banned_mass.detach().cpu().item()))

        def _mask_invalid_tokens(raw_logits):
            vocab_size = raw_logits.size(-1)
            if resolved_valid_vocab_size is None or resolved_valid_vocab_size >= vocab_size:
                return raw_logits
            masked = raw_logits.clone()
            masked[:, resolved_valid_vocab_size:] = float("-inf")
            return masked

        def _mask_banned_tokens(raw_logits):
            if banned_token_tensor is None:
                return raw_logits
            masked = raw_logits.clone()
            masked.index_fill_(dim=-1, index=banned_token_tensor, value=float("-inf"))
            return masked

        def _apply_repetition_penalty(raw_logits):
            if repetition_penalty == 1.0:
                return raw_logits

            penalized = raw_logits.clone()
            for batch_idx, token_ids in enumerate(seen_token_ids):
                if not token_ids:
                    continue
                previous_tokens = torch.tensor(
                    list(token_ids), device=raw_logits.device, dtype=torch.long
                )
                token_logits = penalized[batch_idx, previous_tokens]
                penalized[batch_idx, previous_tokens] = torch.where(
                    token_logits < 0,
                    token_logits * repetition_penalty,
                    token_logits / repetition_penalty,
                )
            return penalized

        def _filter_logits(raw_logits):
            _record_invalid_mass(raw_logits)
            _record_banned_mass(raw_logits)
            filtered = _apply_repetition_penalty(raw_logits)
            filtered = _mask_invalid_tokens(filtered)
            filtered = _mask_banned_tokens(filtered)
            filtered = filtered / max(_adaptive_temp(filtered, temperature), 1e-6)

            vocab_size = filtered.size(-1)
            sampling_vocab_size = vocab_size
            if resolved_valid_vocab_size is not None:
                sampling_vocab_size = min(sampling_vocab_size, resolved_valid_vocab_size)
            effective_top_k = min(max(int(top_k), 0), sampling_vocab_size)
            if effective_top_k > 0:
                top_k_logits, top_k_indices = torch.topk(filtered, effective_top_k)
                if top_p < 1.0:
                    sorted_logits, sorted_order = torch.sort(top_k_logits, descending=True, dim=-1)
                    sorted_indices = top_k_indices.gather(dim=-1, index=sorted_order)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask = cumulative_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    filtered = torch.full_like(filtered, float("-inf"))
                    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
                    return filtered

                masked = torch.full_like(filtered, float("-inf"))
                masked.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
                return masked

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                filtered = torch.full_like(filtered, float("-inf"))
                filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

            return filtered

        def _sample_next(raw_logits):
            next_logits = _filter_logits(raw_logits)
            probs = F.softmax(next_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            if finished.any():
                sampled = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(sampled, resolved_eos_token_id),
                    sampled,
                )
            return sampled

        def _remember(next_token):
            for batch_idx, tok_id in enumerate(next_token.squeeze(1).tolist()):
                if bool(finished[batch_idx]):
                    continue
                seen_token_ids[batch_idx].add(int(tok_id))

        def _check_stop(next_token: torch.Tensor) -> bool:
            for batch_idx, tok_id in enumerate(next_token.squeeze(1).tolist()):
                if bool(finished[batch_idx]):
                    continue

                should_stop = tok_id == resolved_eos_token_id
                recent_ids[batch_idx].append(int(tok_id))
                if len(recent_ids[batch_idx]) > 4:
                    recent_ids[batch_idx].pop(0)
                if (
                    assistant_stop_token_id is not None
                    and tok_id == assistant_stop_token_id
                    and tokenizer is not None
                ):
                    decoded = tokenizer.decode(recent_ids[batch_idx])
                    should_stop = should_stop or (assistant_end in decoded)

                if should_stop:
                    finished[batch_idx] = True

            return bool(finished.all().item())

        raw_logits = logits[:, -1, :]
        next_token = _sample_next(raw_logits)
        generated_tokens.append(next_token)
        _remember(next_token)

        for _ in range(1, max_new_tokens):
            if _check_stop(next_token):
                break
            logits = self.forward(
                next_token,
                use_cache=True,
                past_key_values=past_key_values,
                last_logits_only=True,
            )

            raw_logits = logits[:, -1, :]
            next_token = _sample_next(raw_logits)
            generated_tokens.append(next_token)
            _remember(next_token)

        if diagnostics is not None:
            diagnostics.update({
                "valid_vocab_size": resolved_valid_vocab_size,
                "banned_token_count": len(resolved_banned_token_ids),
                "banned_mass_steps": len(banned_masses),
                "banned_mass_mean": (
                    sum(banned_masses) / len(banned_masses) if banned_masses else 0.0
                ),
                "banned_mass_max": max(banned_masses) if banned_masses else 0.0,
                "invalid_mass_steps": len(invalid_masses),
                "invalid_mass_mean": (
                    sum(invalid_masses) / len(invalid_masses) if invalid_masses else 0.0
                ),
                "invalid_mass_max": max(invalid_masses) if invalid_masses else 0.0,
            })

        return torch.cat([prompt_ids] + generated_tokens, dim=1)

    def get_sparsity_stats(self) -> dict:
        """Return sparsity statistics for debugging and reporting."""
        stats = {}
        with torch.no_grad():
            dummy = torch.randint(0, 100, (1, 32))
            spikes, _ = self.spike_encoder(dummy)
            stats['encoder_sparsity'] = 1.0 - spikes.mean().item()
            for i, block in enumerate(self.blocks):
                spikes, _ = block(spikes, spikes)
                stats[f'block_{i}_sparsity'] = 1.0 - spikes.mean().item()
        return stats
