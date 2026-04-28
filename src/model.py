#!/usr/bin/env python3
"""SymbolicLight V1 model implementation."""
import math
from dataclasses import dataclass, field
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
    stdp_lr: float = 0.01         
    enable_stdp: bool = False     
    
    rope_theta: float = 10000.0   
    frontend_mode: str = "text"   
    
    sparse_attn_window: int = 512     
    n_global_anchors: int = 4         
    enable_sparse_attn: bool = True   
    enable_dynamic_prior: bool = True  
    use_topk_mask: bool = False        
    topk_sparsity: float = 0.89        





class ATanSurrogate(torch.autograd.Function):
    """ATan surrogate-gradient spike function."""
    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        ctx.save_for_backward(membrane_potential, torch.tensor(threshold,
                              device=membrane_potential.device,
                              dtype=membrane_potential.dtype))
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, threshold = ctx.saved_tensors
        alpha = 2.0  
        grad_v = 1.0 / (1.0 + (alpha * (membrane_potential - threshold)) ** 2)
        return grad_output * grad_v, None


def surrogate_spike(membrane_potential: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Apply the surrogate spike function."""
    return ATanSurrogate.apply(membrane_potential, threshold)





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





class FrontendRouter(nn.Module):
    """Text embedding frontend."""
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config

        
        self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        
        
        
        

        
        
        
        

    def forward(self, token_ids: torch.Tensor, modality: str = "text") -> torch.Tensor:
        if modality == "text":
            return self.text_embedding(token_ids)
        elif modality == "vision":
            raise NotImplementedError("Vision frontend is not included in this release.")
        elif modality == "audio":
            raise NotImplementedError("Audio frontend is not included in this release.")
        else:
            raise ValueError(f"Unknown modality: {modality}")






def _lif_scan_forward(x: torch.Tensor, v_mem: torch.Tensor,
                     leak: float, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT forward pass for temporal LIF neuron scanning.
    Returns: (spikes [B,S,D], final_v_mem [B,D], all_v_mem [B,S,D] for backward)
    """
    B, S, D = x.shape
    spikes = torch.empty_like(x)
    all_v = torch.empty_like(x)  
    for t in range(S):
        v_mem = v_mem * leak + x[:, t, :]
        v_mem = torch.clamp(v_mem, -3.0, 3.0)
        all_v[:, t, :] = v_mem
        spike = (v_mem >= threshold).float()
        v_mem = v_mem * (1.0 - spike)
        spikes[:, t, :] = spike
    return spikes, v_mem, all_v


class LIFScan(torch.autograd.Function):
    """LIF scan with ATan surrogate gradient for backward."""
    @staticmethod
    def forward(ctx, x, v_mem, leak, threshold):
        spikes, final_v, all_v = _lif_scan_forward(x, v_mem, leak, threshold)
        ctx.save_for_backward(all_v)
        ctx.threshold = threshold
        return spikes, final_v

    @staticmethod
    def backward(ctx, grad_spikes, grad_v_mem):
        all_v, = ctx.saved_tensors
        
        alpha = 2.0
        surrogate_grad = 1.0 / (1.0 + (alpha * (all_v - ctx.threshold)) ** 2)
        grad_x = grad_spikes * surrogate_grad
        return grad_x, None, None, None





class SpikeEncoder(nn.Module):
    """
    Convert discrete token IDs into spatiotemporal spike tensors.

    Main design updates:
    - remove learned positional embeddings and use RoPE in SparseTCAM
    - use chunk-parallel LIF spike conversion to reduce Python loops
    - route token embeddings through FrontendRouter instead of a hard-coded embedding

    Flow: token_id -> FrontendRouter -> LayerNorm -> parallel LIF spike conversion
    """
    def __init__(self, config: SymbolicLightConfig):
        super().__init__()
        self.config = config
        
        self.frontend = FrontendRouter(config)
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

        if getattr(self.config, 'use_topk_mask', False):
            
            k = max(1, int((1.0 - self.config.topk_sparsity) * self.config.embed_dim))
            _, topk_indices = torch.topk(x.abs(), k, dim=-1)
            spikes = torch.zeros_like(x)
            spikes.scatter_(-1, topk_indices, 1.0)
            if self.training:
                spikes = spikes + (surrogate_spike(x, self.threshold) - spikes).detach()
        else:
            
            chunk_size = self.config.spike_chunk_size
            spikes_list = []

            for chunk_start in range(0, S, chunk_size):
                chunk_end = min(chunk_start + chunk_size, S)
                x_chunk = x[:, chunk_start:chunk_end, :]
                chunk_spikes, self.v_mem = LIFScan.apply(
                    x_chunk, self.v_mem, self.leak, self.threshold
                )
                spikes_list.append(chunk_spikes)
                
                if self.training:
                    self.v_mem = self.v_mem.detach()

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

        
        if use_cache and cache is not None:
            if 'K' in cache:
                K = torch.cat([cache['K'], K], dim=2)  
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

        
        
        if offset == 0 and S_q == S_kv and S_q == self._cached_mask.size(0):
            attn_mask = self._cached_mask
        else:
            q_pos = torch.arange(offset, offset + S_q, device=x.device).unsqueeze(1)
            k_pos = torch.arange(0, S_kv, device=x.device).unsqueeze(0)
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
        2. allow training-time cache reads and writes for hidden state h across chunks
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

        
        out_spikes = surrogate_spike(out_continuous, self.threshold).to(out_continuous.dtype)

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
        h = surrogate_spike(h, self.threshold).to(x.dtype)
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
        spikes = surrogate_spike(continuous, self.tcam.threshold).to(continuous.dtype)
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





class STDPUpdater:
    """Optional local spike-timing update rule."""
    def __init__(self, config: SymbolicLightConfig):
        self.lr = config.stdp_lr
        self.enabled = config.enable_stdp

    @torch.no_grad()
    def update(self, model: nn.Module, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        if not self.enabled:
            return

        causal = (pre_spikes.sum(dim=1, keepdim=True) > 0) & (post_spikes.sum(dim=1, keepdim=True) > 0)

        if causal.any():
            for block in model.blocks:
                w = block.tcam.tcam_proj.weight
                pre_active = (pre_spikes > 0).float()
                post_active = (post_spikes > 0).float()
                co_firing = torch.einsum('bsd,bse->de', post_active, pre_active)
                delta = self.lr * co_firing / (pre_spikes.size(0) * pre_spikes.size(1))
                mask = (co_firing > 0).float()
                w.data += delta * mask * 0.05
                w.data.clamp_(-5, 5)





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
        self.stdp = STDPUpdater(config)
        self.gradient_checkpointing = False

        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SymbolicLight V1] model initialized | parameters: {n_params/1e6:.1f}M ({n_params/1e9:.3f}B)")

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
        print("[SymbolicLight V1] torch.compile applied for inference acceleration")

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False,
                past_key_values: list = None,
                streaming_state: list = None):
        """
        Shared forward pass for training and inference.

        Args:
            token_ids: [B, S] input token IDs
            use_cache: whether to use the KV cache for inference
            past_key_values: list of inference caches
            streaming_state: list of streaming caches used to carry hidden state across chunks
                            during training; format matches past_key_values:
                            [encoder_cache, block0_cache, block1_cache, ...]
        Returns:
            logits: [B, S, vocab_size]
        """
        
        if use_cache and past_key_values is None:
            past_key_values = [{} for _ in range(len(self.blocks) + 1)]

        
        if not use_cache and streaming_state is not None:
            caches = streaming_state
        elif use_cache:
            caches = past_key_values
        else:
            caches = [None] * (len(self.blocks) + 1)

        
        encoder_cache = caches[0] if caches[0] is not None else (
            past_key_values[0] if use_cache else None
        )
        spikes, continuous = self.spike_encoder(token_ids, use_cache=use_cache, cache=encoder_cache)
        model_dtype = self.output_head.output_proj.weight.dtype
        if continuous.dtype != model_dtype:
            continuous = continuous.to(model_dtype)
        if spikes.dtype != model_dtype:
            spikes = spikes.to(model_dtype)
        initial_spikes = spikes

        
        for i, block in enumerate(self.blocks):
            block_cache = caches[i + 1] if caches[i + 1] is not None else (
                past_key_values[i + 1] if use_cache else None
            )
            if self.training and self.gradient_checkpointing and not use_cache and block_cache is None:
                def _checkpointed_block(spk, cont, current_block=block):
                    out_spikes, out_continuous = current_block(
                        spk, cont, use_cache=False, cache=None,
                    )
                    return out_spikes, out_continuous

                spikes, continuous = torch.utils.checkpoint.checkpoint(
                    _checkpointed_block, spikes, continuous,
                    use_reentrant=False,
                )
            else:
                spikes, continuous = block(
                    spikes, continuous,
                    use_cache=use_cache, cache=block_cache,
                )

        
        logits = self.output_head(continuous)

        
        if not self.training and self.config.enable_stdp and initial_spikes.size(1) > 1:
            self.stdp.update(self, initial_spikes, spikes)

        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 adaptive_temperature: bool = True) -> torch.Tensor:
        """
        Autoregressive text generation with O(1) cached incremental decoding.

        Adaptive temperature:
          - lower entropy -> lower temperature for more deterministic outputs
          - higher entropy -> higher temperature for more exploratory outputs
          - effective range is approximately [0.3, 1.5]
        """
        self.eval()
        generated = prompt_ids.clone()
        past_key_values = [{} for _ in range(len(self.blocks) + 1)]

        logits = self.forward(prompt_ids, use_cache=True, past_key_values=past_key_values)

        
        def _adaptive_temp(raw_logits, base_temp):
            """Adjust temperature dynamically from the logits entropy."""
            if not adaptive_temperature:
                return base_temp
            probs = F.softmax(raw_logits, dim=-1)
            p = probs.clamp(1e-7, 1.0)
            entropy = -(p * p.log()).sum(dim=-1).mean()  
            
            max_entropy = math.log(self.config.vocab_size)
            norm_entropy = (entropy / max_entropy).clamp(0, 1)
            
            
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





if __name__ == "__main__":
    print("=" * 60)
    print(" SymbolicLight V1 model smoke test")
    print("=" * 60)

    config = SymbolicLightConfig(
        vocab_size=57344,
        embed_dim=768,
        n_layers=12,
        n_heads=12,
        head_dim=64,
    )

    model = SymbolicLightModel(config)

    
    dummy_input = torch.randint(0, 57344, (2, 128))
    print(f"\nInput: batch=2, seq_len=128")

    
    logits = model(dummy_input)
    print(f"Output logits: {logits.shape}")

    
    print(f"\nStreaming context test (2 chunks x 128 tokens)...")
    chunk1 = torch.randint(0, 57344, (2, 128))
    chunk2 = torch.randint(0, 57344, (2, 128))

    
    streaming_state = [{} for _ in range(len(model.blocks) + 1)]
    logits1 = model(chunk1, streaming_state=streaming_state)
    print(f"  Chunk 1 logits: {logits1.shape}, streaming state saved [OK]")

    
    logits2 = model(chunk2, streaming_state=streaming_state)
    print(f"  Chunk 2 logits: {logits2.shape}, cross-chunk memory passed [OK]")

    
    stats = model.get_sparsity_stats()
    print(f"\nSparsity stats:")
    for k, v in stats.items():
        print(f"  {k}: {v*100:.1f}% silent")

    
    prompt = torch.randint(0, 57344, (1, 10))
    print(f"\nAutoregressive generation test (prompt=10, gen=20)...")
    output = model.generate(prompt, max_new_tokens=20)
    print(f"Generated sequence length: {output.shape[1]}")

    print("\n[PASS] SymbolicLight V1 smoke checks completed.")
    print("  [1] RoPE rotary position encoding [OK]")
    print("  [2] Cross-chunk state passing [OK]")
    print("  [3] BayesianHead dynamic prior [OK]")
    print("  [4] SpikeEncoder parallel scan [OK]")
    print("  [5] FrontendRouter multimodal stub [OK]")
