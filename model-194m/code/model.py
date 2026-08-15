#!/usr/bin/env python3
"""
SymbolicLight — Spike-Gated Dual-Path Language Model
=====================================================
Architecture components:
  - SpikeEncoder: LIF neuron encoding with chunked sequential processing
  - SparseTCAM: Dual-path sequence mixer (exponential decay + spike-gated attention)
  - BayesianHead: Dynamic context-aware prior network
  - RoPE: Rotary Position Embedding (applied inside attention)
  - EntropyGate: Confidence-based early exit (optional)
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  Configuration
# ============================================================================
@dataclass
class SymbolicLightConfig:
    """SymbolicLight default configuration"""
    vocab_size: int = 48000        # Vocabulary size (SL-BPE bilingual tokenizer)
    embed_dim: int = 768          # Embedding dimension
    n_layers: int = 12            # Number of SymbolicLightBlock layers
    n_heads: int = 12             # Number of "channels" in SparseTCAM
    head_dim: int = 64            # Dimension per channel (embed_dim / n_heads)
    intermediate_dim: int = 4096  # FFN intermediate dimension
    max_seq_len: int = 2048       # Max sequence length (RoPE supports extrapolation beyond this)
    spike_chunk_size: int = 64    # SpikeEncoder chunk size for sequential processing
    dropout: float = 0.1
    # --- SNN-specific parameters ---
    spike_threshold: float = 1.0  # LIF neuron firing threshold
    leak_factor: float = 0.95     # Membrane potential leak factor
    entropy_exit_threshold: float = 0.3  # Entropy gate early exit threshold
    enable_entropy_exit: bool = False  # Whether to enable entropy-based early exit
    # --- RoPE parameters ---
    rope_theta: float = 10000.0   # RoPE base frequency (higher = better extrapolation)
    frontend_mode: str = "text"   # Frontend mode
    # --- Sparse local attention ---
    sparse_attn_window: int = 256     # Sliding window size
    n_global_anchors: int = 4         # Number of global anchor tokens
    enable_sparse_attn: bool = True   # Whether to enable sparse local attention
    enable_dynamic_prior: bool = True  # Whether to enable dynamic Bayesian prior


def hard_spike(membrane_potential: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Apply the inference-time hard spike threshold."""
    return (membrane_potential >= threshold).float()


# ============================================================================
#  RoPE — Rotary Position Embedding (from RoFormer / Llama)
# ============================================================================
class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Advantages over learnable position embeddings:
    1. No hard length limit: no max_seq_len Embedding table, supports length extrapolation
    2. Relative position awareness: encodes relative distance, not absolute position
    3. No learned absolute-position table
    4. Natural decay: influence of distant tokens weakens naturally

    Applied inside SparseLocalAttention after Q/K projection.
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        # Precompute inverse frequencies: theta_i = 1 / (theta^(2i/d))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE rotation to input tensor.
        Args:
            x: [B, S, D] input continuous representation
            offset: position offset (for incremental inference)
        Returns:
            rotated: [B, S, D] rotated representation
        """
        B, S, D = x.shape
        # Generate position sequence
        t = torch.arange(offset, offset + S, device=x.device, dtype=torch.float32)
        # Outer product: [S] x [D//2] -> [S, D//2]
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        # Duplicate to [S, D]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, D]
        cos_emb = emb.cos().unsqueeze(0)  # [1, S, D]
        sin_emb = emb.sin().unsqueeze(0)  # [1, S, D]

        # Rotation: split into two halves, cross-rotate
        x_rotated = torch.cat([
            -x[..., D // 2:],  # negate second half
             x[..., :D // 2],  # keep first half
        ], dim=-1)

        return x * cos_emb + x_rotated * sin_emb


# ============================================================================
#  Text Frontend — Token Embedding
# ============================================================================
class FrontendRouter(nn.Module):
    """Text embedding frontend. Maps token IDs to dense vectors [B, S, embed_dim]."""
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

    def forward(self, token_ids: torch.Tensor, modality: str = "text") -> torch.Tensor:
        """
        Args:
            token_ids: [B, S] token IDs
        Returns:
            embeddings: [B, S, embed_dim]
        """
        return self.text_embedding(token_ids)


# ============================================================================
#  SpikeEncoder — LIF Neuron Encoding with Chunked Sequential Processing
# ============================================================================
class SpikeEncoder(nn.Module):
    """
    Converts discrete token IDs into spatiotemporal spike tensors.

    Pipeline: token_id -> Embedding -> LayerNorm -> LIF chunked spiking

    RoPE is NOT applied here; it is applied inside SparseLocalAttention
    after Q/K projection to keep the residual stream clean.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        # Text embedding frontend
        self.frontend = FrontendRouter(config)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.threshold = config.spike_threshold
        self.leak = config.leak_factor

        # No learnable position encoding; RoPE is applied in SparseTCAM.

        # Membrane potential (dynamically managed per batch)
        self.v_mem = None  # Not in state_dict to avoid shape mismatch

    def _init_membrane(self, shape: torch.Size, device: torch.device):
        """Initialize/reset membrane potential."""
        self.v_mem = torch.zeros(shape, device=device)

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False,
                cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len]
            use_cache: if True, perform O(1) incremental forward
            cache: cache dictionary
        Returns:
            spikes: [batch, seq_len, embed_dim]  sparse 0/1 binary spikes
            continuous: [batch, seq_len, embed_dim]  continuous representation (for residual)
        """
        B, S = token_ids.shape

        # Read cached state
        if use_cache and cache is not None:
            if 'v_mem' not in cache:
                cache['v_mem'] = torch.zeros(B, self.config.embed_dim, device=token_ids.device)
            if 'seq_len' not in cache:
                cache['seq_len'] = 0
            self.v_mem = cache['v_mem']
            cache['seq_len'] += S
        else:
            self._init_membrane((B, self.config.embed_dim), token_ids.device)

        # Embedding (no position encoding; RoPE applied downstream)
        x = self.frontend(token_ids)
        x = self.norm(x)

        chunk_size = self.config.spike_chunk_size
        spikes_list = []

        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            x_chunk = x[:, chunk_start:chunk_end, :]
            chunk_len = chunk_end - chunk_start
            chunk_spikes = []
            for t in range(chunk_len):
                self.v_mem = self.v_mem * self.leak + x_chunk[:, t, :]
                self.v_mem = torch.clamp(self.v_mem, min=-3.0, max=3.0)
                spike = hard_spike(self.v_mem, self.threshold)
                self.v_mem = self.v_mem * (1.0 - spike)
                chunk_spikes.append(spike)
            spikes_list.append(torch.stack(chunk_spikes, dim=1))

        spikes = torch.cat(spikes_list, dim=1)

        # Write to cache
        if use_cache and cache is not None:
            cache['v_mem'] = self.v_mem.detach()

        return spikes, x  # Return spikes and continuous representation


# ============================================================================
#  SparseLocalAttention — Spike-Gated Local Attention (Sliding Window + Anchors)
# ============================================================================
class SparseLocalAttention(nn.Module):
    """
    Computes attention only among active spike positions with sliding window + global anchors.

    Complexity:
    - Full Attention: all S positions attend to all -> O(S^2)
    - This module: only ~9% positions are active, within local window -> O(S * k * w)
      where k = active ratio (~0.09), w = window size (256)
    - Global anchors: first N tokens can attend to all positions for global semantic grounding

    Relationship with exponential decay path:
    - Decay path: compress history into fixed-size hidden state -> historical context compression
    - Attention path: precise focus on recent key positions -> short-range fine-grained reasoning
    - Both paths fused via learnable gating
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.window_size = config.sparse_attn_window
        self.n_global_anchors = config.n_global_anchors
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

        # Q/K/V projections (independent from TCAM projection)
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # RoPE applied inside attention after Q/K projection
        self.rope = RotaryPositionEncoding(config.head_dim, theta=config.rope_theta)

    def forward(self, x: torch.Tensor, spike_mask: torch.Tensor,
                offset: int = 0, use_cache: bool = False, cache: dict = None) -> torch.Tensor:
        """
        Args:
            x: [B, S_q, D] continuous representation (unrotated; RoPE applied internally to Q/K)
            spike_mask: [B, S_q] boolean mask, True = active spike at this position
            offset: RoPE position offset (for incremental inference)
            use_cache: whether to use KV Cache (for incremental inference)
            cache: KV Cache dictionary
        Returns:
            attn_out: [B, S_q, D] sparse attention output (inactive positions are zeroed)
        """
        B, S_q, D = x.shape

        # 1. Q/K/V projection + multi-head reshape
        # Must transpose(1,2) so B and H are adjacent, then contiguous() + reshape
        # Otherwise [B,S,H,d] reshape to [B*H,S,d] would mix S and H dimensions
        Q = self.q_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, S_q, d]
        K = self.k_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S_q, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Apply RoPE after Q/K projection (memory-safe reshape)
        Q = self.rope(Q.contiguous().view(B * self.n_heads, S_q, self.head_dim), offset=offset)
        Q = Q.view(B, self.n_heads, S_q, self.head_dim)
        K = self.rope(K.contiguous().view(B * self.n_heads, S_q, self.head_dim), offset=offset)
        K = K.view(B, self.n_heads, S_q, self.head_dim)

        # 3. KV Cache: cache historical K/V and spike masks during incremental inference
        if use_cache and cache is not None:
            if 'K' in cache:
                K = torch.cat([cache['K'], K], dim=2)  # [B, H, S_kv, d]
                V = torch.cat([cache['V'], V], dim=2)
                spike_mask_kv = torch.cat([cache['spike_mask'], spike_mask], dim=1)
            else:
                spike_mask_kv = spike_mask
            cache['K'] = K.detach()
            cache['V'] = V.detach()
            cache['spike_mask'] = spike_mask_kv.detach()
        else:
            spike_mask_kv = spike_mask

        S_kv = K.size(2)

        # 4. Build attention mask (compatible with asymmetric S_q x S_kv for incremental inference)
        q_pos = torch.arange(offset, offset + S_q, device=x.device).unsqueeze(1)  # [S_q, 1]
        k_pos = torch.arange(0, S_kv, device=x.device).unsqueeze(0)  # [1, S_kv]

        causal = q_pos >= k_pos  # [S_q, S_kv]
        window = (q_pos - k_pos).abs() <= self.window_size
        anchors = k_pos < self.n_global_anchors
        attn_mask = causal & (window | anchors)  # [S_q, S_kv]
        # Sparsity enhancement: only active spike positions participate as Key/Value
        spike_key_mask = spike_mask_kv.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_kv]
        full_mask = attn_mask.unsqueeze(0).unsqueeze(0) & spike_key_mask  # [B, 1, S_q, S_kv]

        # 5. Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, S_q, S_kv]
        scores = scores.masked_fill(~full_mask, float('-inf'))
        all_masked = full_mask.sum(dim=-1, keepdim=True) == 0
        scores = scores.masked_fill(all_masked, 0.0)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(all_masked, 0.0)

        # 6. Weighted sum and reshape
        attn_out = torch.matmul(attn_weights, V)  # [B, H, S_q, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, D)  # [B, S_q, D]

        # Zero out inactive positions (only current step's spike mask)
        attn_out = attn_out * spike_mask.unsqueeze(-1).float()

        return attn_out


# ============================================================================
#  SparseTCAM — Dual-Path Sequence Mixer (Decay + Spike-Gated Attention)
# ============================================================================
class SparseTCAM(nn.Module):
    """
    Dual-path architecture:
    - Decay path: exponential decay aggregation -> historical context compression
    - Attention path: sparse local attention -> short-range fine-grained reasoning
    - Learnable gating fuses both paths
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.threshold = config.spike_threshold
        self.leak = config.leak_factor
        self.enable_sparse_attn = config.enable_sparse_attn

        # TCAM weight matrix (content-addressable memory)
        self.tcam_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Learnable exponential decay factor — independent per head
        self.decay_raw = nn.Parameter(torch.full((config.n_heads,), 3.0))

        # RoPE is applied inside SparseLocalAttention (after Q/K projection)
        # to keep the residual stream clean

        # Sparse local attention + gated fusion
        if self.enable_sparse_attn:
            self.sparse_attn = SparseLocalAttention(config)
            # Learnable gate: determines decay vs attention mixing ratio
            # sigmoid(0) = 0.5, initially 50/50 split
            self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, spikes: torch.Tensor, continuous: torch.Tensor,
                use_cache: bool = False, cache: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-path forward:
        1. RoPE applied inside SparseLocalAttention (after Q/K projection)
        2. Cached decoding supports cross-chunk hidden state passing
        """
        B, S, D = spikes.shape

        # RoPE offset management (passed to SparseLocalAttention)
        offset = 0
        if use_cache and cache is not None:
            offset = cache.get('rope_offset', 0)
            cache['rope_offset'] = offset + S

        # 1. Spike mask routing: compute only at positions with active spikes
        spike_energy = spikes.sum(dim=-1)  # [B, S]
        active_mask = (spike_energy > 0).unsqueeze(-1).float()  # [B, S, 1]

        # 2. Content-addressable memory lookup: spikes x TCAM weights
        tcam_out = self.tcam_proj(spikes * active_mask)

        # 3. Multi-head channel fusion
        tcam_out = tcam_out.view(B, S, self.n_heads, self.head_dim)

        # 4. Temporal context aggregation: learnable exponential decay
        decay = torch.sigmoid(self.decay_raw)  # [n_heads] -> (0, 1)

        # Cross-chunk state passing for cached decoding.
        if cache is not None:
            if 'h' not in cache:
                cache['h'] = torch.zeros(B, self.n_heads, self.head_dim, device=spikes.device)
            h = cache['h']
        else:
            h = torch.zeros(B, self.n_heads, self.head_dim, device=spikes.device)

        if use_cache and cache is not None and S == 1:
            # Incremental O(1) inference mode
            h = decay.view(1, self.n_heads, 1) * h + (1 - decay.view(1, self.n_heads, 1)) * tcam_out[:, 0]
            cache['h'] = h.detach()
            context = h.unsqueeze(1)  # [B, 1, n_heads, head_dim]
        else:
            # Convolution implementation for prompt prefill.
            powers = torch.arange(S - 1, -1, -1, dtype=torch.float32, device=spikes.device)
            kernel = ((decay.view(-1, 1) ** powers.view(1, -1)) * (1 - decay).view(-1, 1)).unsqueeze(1)
            tcam_out_trans = tcam_out.permute(0, 3, 2, 1).reshape(-1, self.n_heads, S)
            tcam_out_pad = F.pad(tcam_out_trans, (S - 1, 0))
            out = F.conv1d(tcam_out_pad, kernel, groups=self.n_heads)
            context = out.view(-1, self.head_dim, self.n_heads, S).permute(0, 3, 2, 1)

            # Fuse historical hidden state from previous chunk (cross-window memory passing)
            powers_fwd = torch.arange(1, S + 1, dtype=torch.float32, device=spikes.device).view(1, S, 1, 1)
            decay_t = decay.view(1, 1, self.n_heads, 1) ** powers_fwd
            context = context + h.unsqueeze(1) * decay_t

            # Save last timestep's hidden state for the next chunk
            if cache is not None:
                cache['h'] = context[:, -1, :, :].detach()

        # 5. Merge channels
        decay_output = context.reshape(B, S, D)

        # Sparse local attention path
        if self.enable_sparse_attn:  # KV Cache ensures S=1 also works
            spike_mask = (spikes.sum(dim=-1) > 0)  # [B, S]
            # Separate sub-dict to avoid key collision with hidden state 'h'
            attn_cache = cache.setdefault('attn', {}) if cache is not None else None
            attn_output = self.sparse_attn(
                continuous, spike_mask, offset=offset,
                use_cache=use_cache, cache=attn_cache
            )

            # Gated fusion: output = gate * attn + (1 - gate) * decay
            gate = torch.sigmoid(self.attn_gate)  # (0, 1)
            output = gate * attn_output + (1 - gate) * decay_output
        else:
            output = decay_output

        output = self.out_proj(self.dropout(output))

        # 6. Residual connection + LayerNorm
        # Use unrotated continuous to keep residual stream clean
        out_continuous = self.norm(continuous + output)

        # 7. LIF spiking output
        out_spikes = hard_spike(out_continuous, self.threshold)

        return out_spikes, out_continuous


# ============================================================================
#  EntropyGate — Confidence-Based Early Exit
# ============================================================================
class EntropyGate(nn.Module):
    """
    Entropy-based early exit gate.
    Low entropy = model is confident -> can exit early without running all layers.
    High entropy = model is uncertain -> continue to deeper layers.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.threshold = config.entropy_exit_threshold

    def forward(self, continuous: torch.Tensor, exit_head: nn.Module) -> Tuple[torch.Tensor, bool]:
        exit_logits = exit_head(continuous)

        probs = F.softmax(exit_logits[:, -1, :], dim=-1)
        p = probs.clamp(1e-7, 1.0)
        entropy = -(p * p.log()).sum(dim=-1).mean()
        should_exit = entropy.item() < self.threshold

        return exit_logits, should_exit


# ============================================================================
#  SpikingFeedForward — FFN with LIF Spike Activation
# ============================================================================
class SpikingFeedForward(nn.Module):
    """
    Two-layer MLP replacing Transformer FFN.
    Key difference: intermediate layer uses LIF spike activation instead of GELU/ReLU.
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
        h = hard_spike(h, self.threshold)
        h = self.down(self.dropout(h))
        return self.norm(residual + h)


# ============================================================================
#  SymbolicLightBlock — Single Layer Block
# ============================================================================
class SymbolicLightBlock(nn.Module):
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.tcam = SparseTCAM(config)
        self.ffn = SpikingFeedForward(config)
        self.entropy_gate = EntropyGate(config)

    def forward(self, spikes, continuous, exit_head, use_cache=False, cache=None,
                need_exit_logits=True):
        spikes, continuous = self.tcam(spikes, continuous, use_cache=use_cache, cache=cache)
        continuous = self.ffn(continuous)
        spikes = hard_spike(continuous, self.tcam.threshold)
        if need_exit_logits:
            exit_logits, should_exit = self.entropy_gate(continuous, exit_head)
        else:
            exit_logits, should_exit = None, False
        return spikes, continuous, should_exit, exit_logits


# ============================================================================
#  BayesianHead — Dynamic Context-Aware Prior Output Head
# ============================================================================
class BayesianHead(nn.Module):
    """
    Dynamic context-aware Bayesian output head.

    Bayesian formulation:
      log P(word|context) = log P(context|word) + log P(word|context_summary)
                           ^ likelihood (output_proj)  ^ dynamic prior (prior_net)

    The prior network generates context-dependent word priors, adapting to
    the current topic (e.g., boosting cooking-related words when discussing recipes).
    This is more precise than a static learnable frequency bias.
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.use_dynamic_prior = getattr(config, 'enable_dynamic_prior', True)

        if self.use_dynamic_prior:
            bottleneck_dim = config.embed_dim // 4  # 192 for embed_dim=768
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
        log_likelihood = self.output_proj(continuous)  # [B, S, V]

        if self.use_dynamic_prior:
            dynamic_prior = self.prior_net(continuous)  # [B, S, V]
            logits = log_likelihood + 0.1 * dynamic_prior
        else:
            logits = log_likelihood + 0.1 * self.log_prior

        return logits


# ============================================================================
#  Full Model
# ============================================================================
class SymbolicLightModel(nn.Module):
    """
    SymbolicLight: Spike-Gated Dual-Path Language Model

    Architecture:
    - RoPE rotary position encoding for length extrapolation
    - Cross-chunk state passing for historical state transfer
    - BayesianHead with dynamic context-aware prior
    - SpikeEncoder with chunked sequential LIF processing
    - Dual-path SparseTCAM (decay + spike-gated attention)
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        self.spike_encoder = SpikeEncoder(config)
        self.blocks = nn.ModuleList([
            SymbolicLightBlock(config) for _ in range(config.n_layers)
        ])
        self.output_head = BayesianHead(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compile_for_inference(self):
        """Apply torch.compile to critical submodules for inference acceleration."""
        self.spike_encoder = torch.compile(self.spike_encoder, mode='reduce-overhead')
        for block in self.blocks:
            block.tcam = torch.compile(block.tcam, mode='reduce-overhead')
            block.ffn = torch.compile(block.ffn, mode='reduce-overhead')

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False,
                past_key_values: list = None):
        """Inference-only forward pass."""
        if use_cache and past_key_values is None:
            past_key_values = [{} for _ in range(len(self.blocks) + 1)]

        caches = past_key_values if use_cache else [None] * (len(self.blocks) + 1)

        # Spike encoding
        encoder_cache = caches[0] if caches[0] is not None else (
            past_key_values[0] if use_cache else None
        )
        spikes, continuous = self.spike_encoder(token_ids, use_cache=use_cache, cache=encoder_cache)
        need_exit = self.config.enable_entropy_exit and not use_cache

        for i, block in enumerate(self.blocks):
            block_cache = caches[i + 1] if caches[i + 1] is not None else (
                past_key_values[i + 1] if use_cache else None
            )
            spikes, continuous, should_exit, exit_logits = block(
                spikes, continuous, self.output_head,
                use_cache=use_cache, cache=block_cache,
                need_exit_logits=need_exit
            )

            if should_exit and self.config.enable_entropy_exit:
                return exit_logits

        # Bayesian output (final layer)
        logits = self.output_head(continuous)

        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 adaptive_temperature: bool = True) -> torch.Tensor:
        """
        Autoregressive text generation with O(1) incremental inference.

        Adaptive temperature:
          - Low entropy (confident) -> lower temperature (reduce hallucination)
          - High entropy (uncertain) -> higher temperature (encourage exploration)
          - Temperature range [0.3, 1.5]
        """
        self.eval()
        generated = prompt_ids.clone()
        past_key_values = [{} for _ in range(len(self.blocks) + 1)]

        logits = self.forward(prompt_ids, use_cache=True, past_key_values=past_key_values)

        # Adaptive temperature computation
        def _adaptive_temp(raw_logits, base_temp):
            """Dynamically adjust temperature based on logits entropy."""
            if not adaptive_temperature:
                return base_temp
            probs = F.softmax(raw_logits, dim=-1)
            p = probs.clamp(1e-7, 1.0)
            entropy = -(p * p.log()).sum(dim=-1).mean()  # scalar
            # Normalize to [0, 1]: max entropy = log(vocab_size)
            max_entropy = math.log(self.config.vocab_size)
            norm_entropy = (entropy / max_entropy).clamp(0, 1)
            # High entropy (uncertain) -> lower temp toward argmax to prevent hallucination
            # Low entropy (confident) -> keep base temp for diversity
            temp = max(0.1, base_temp - norm_entropy.item() * (base_temp - 0.1))
            return temp

        raw_logits = logits[:, -1, :]
        temp = _adaptive_temp(raw_logits, temperature)
        next_logits = raw_logits / temp
        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, top_k)
            min_top_k = top_k_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < min_top_k] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        for _ in range(1, max_new_tokens):
            if next_token.item() == 2:
                break
            logits = self.forward(next_token, use_cache=True, past_key_values=past_key_values)

            raw_logits = logits[:, -1, :]
            temp = _adaptive_temp(raw_logits, temperature)
            next_logits = raw_logits / temp

            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k)
                min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_top_k] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
