#!/usr/bin/env python3
"""
MT3Model PyTorch Implementation

Production-ready PyTorch implementation of MT3 based on T5 architecture.
Designed for compatibility with converted T5X checkpoints.

Model Architecture:
- T5 encoder-decoder with 8 layers each
- d_model: 512
- vocab_size: 1536
- num_heads: 8
- d_ff: 1024
- Parameter count: ~45.8M parameters

Key Features:
- T5-style relative position bias
- Shared embedding layer
- Cross-attention in decoder
- RMSNorm (T5 standard)
- Compatible parameter naming
"""

import math
import warnings
from typing import Optional, Dict, Tuple, Any, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MT3Config:
    """Configuration class for MT3Model."""

    # Model architecture
    vocab_size: int = 1536
    d_model: int = 512
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    num_heads: int = 8
    d_ff: int = 1024
    d_kv: int = 48  # MT3 uses 48 (inner_dim 384 = 8 heads × 48)

    # Training parameters
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6

    # Position encoding
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    # Special tokens
    decoder_start_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 0

    # Generation parameters
    max_length: int = 1024

    def __post_init__(self):
        """Validate configuration parameters."""
        # Note: For MT3, d_kv (48) is intentionally different from d_model//num_heads (64)
        # This is part of the original MT3 architecture design
        expected_inner_dim = self.num_heads * self.d_kv
        if expected_inner_dim != 384:
            warnings.warn(f"inner_dim ({expected_inner_dim}) differs from MT3 standard (384 = 8 × 48)")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (T5 standard)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply RMS normalization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MT3DenseActivation(nn.Module):
    """Dense layer with gated activation (T5 standard)."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = F.relu

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass with gated activation."""
        hidden_gelu = self.activation(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        return self.wo(hidden_states)


class MT3LayerFF(nn.Module):
    """Feed-forward layer for MT3."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.DenseReluDense = MT3DenseActivation(config)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through feed-forward layer."""
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class MT3Attention(nn.Module):
    """Multi-head attention with relative position bias (T5 style)."""

    def __init__(self, config: MT3Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = hasattr(self, 'is_decoder') and self.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Linear projections
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # Relative position bias
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Compute relative position bucket for bias."""
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Other half of buckets are for logarithmically bigger bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute relative position bias."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        layer_head_mask: Optional[Tensor] = None,
        query_length: Optional[int] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Forward pass for multi-head attention."""
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length
        if past_key_value is not None:
            real_seq_length += past_key_value[0].shape[-2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """Projection to heads."""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """Reshape back from heads."""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """Project hidden states correctly to match expected shapes."""
            if key_value_states is None:
                # Self-attention
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # Cross-attention
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # Self-attention
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=-2)
                else:
                    # Cross-attention
                    hidden_states = past_key_value
            return hidden_states

        # Compute query, key, value
        query_states = shape(self.q(hidden_states))
        key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)

        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # Recalculate position_bias when using cache (past_key_value) since shapes change each step
        if position_bias is None or (past_key_value is not None and self.is_decoder):
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # If this is a decoder and we have past key/values, only use position bias for the last token
            if past_key_value is not None and self.is_decoder:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply head mask if provided
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class MT3LayerSelfAttention(nn.Module):
    """Self-attention layer for MT3."""

    def __init__(self, config: MT3Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = MT3Attention(config, has_relative_attention_bias)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Forward pass for self-attention layer."""
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class MT3LayerCrossAttention(nn.Module):
    """Cross-attention layer for MT3 decoder."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.EncDecAttention = MT3Attention(config, has_relative_attention_bias=False)
        self.EncDecAttention.is_decoder = True
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        query_length: Optional[int] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Forward pass for cross-attention layer."""
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )

        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class MT3Block(nn.Module):
    """Single transformer block for MT3 encoder."""

    def __init__(self, config: MT3Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(MT3LayerSelfAttention(config, has_relative_attention_bias))
        self.layer.append(MT3LayerFF(config))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Forward pass for encoder block."""
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]

        hidden_states = self.layer[1](hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


class MT3BlockDecoder(nn.Module):
    """Single transformer block for MT3 decoder."""

    def __init__(self, config: MT3Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(MT3LayerSelfAttention(config, has_relative_attention_bias))
        self.layer.append(MT3LayerCrossAttention(config))
        self.layer.append(MT3LayerFF(config))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_decoder_position_bias: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        cross_attn_layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Forward pass for decoder block."""
        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        # Self-attention
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]

        # Cross-attention
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            if use_cache:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Feed forward
        hidden_states = self.layer[2](hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


class MT3Stack(nn.Module):
    """Base class for encoder and decoder stacks."""

    def __init__(self, config: MT3Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_decoder = False
        self.block = nn.ModuleList()
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[Tensor], Dict[str, Any]]:
        """Forward pass for the stack."""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # Initialize past_key_values if needed
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.block)

        hidden_states = self.dropout(inputs_embeds)

        position_bias = None
        encoder_decoder_position_bias = None

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.is_decoder else None

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            cross_attn_layer_head_mask = cross_attn_head_mask[i] if cross_attn_head_mask is not None else None

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.is_decoder:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    layer_head_mask=layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                presents = presents + (layer_outputs[1],)

            # Update position bias (only for first layer and only if not using cache with past_key_values)
            # When using cache, position_bias must be recalculated each step due to changing dimensions
            if i == 0 and not (use_cache and past_key_values is not None):
                position_bias = layer_outputs[2 if use_cache else 1]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[3 if use_cache else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-2 if self.is_decoder else -1],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[-1],)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ] if v is not None
            )

        return {
            'last_hidden_state': hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
            'cross_attentions': all_cross_attentions,
        }


class MT3Encoder(MT3Stack):
    """MT3 encoder stack."""

    def __init__(self, config: MT3Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        # Input projection for continuous audio features (mel spectrogram → d_model)
        # MT3 checkpoint uses 512 mel bins, projects to d_model (512)
        self.continuous_inputs_projection = nn.Linear(512, config.d_model, bias=False)

        for i in range(config.num_encoder_layers):
            self.block.append(MT3Block(config, has_relative_attention_bias=(i == 0)))

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        **kwargs
    ):
        """Forward pass for encoder with continuous audio input projection."""
        # If inputs_embeds provided (audio features), project from mel_bins to d_model
        if inputs_embeds is not None and inputs_embeds.size(-1) == 512:
            # Audio features: [batch, seq_len, 512] → [batch, seq_len, 512]
            inputs_embeds = self.continuous_inputs_projection(inputs_embeds)

        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)


class MT3Decoder(MT3Stack):
    """MT3 decoder stack."""

    def __init__(self, config: MT3Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.is_decoder = True

        for i in range(config.num_decoder_layers):
            self.block.append(MT3BlockDecoder(config, has_relative_attention_bias=(i == 0)))


class MT3Model(nn.Module):
    """
    MT3 Model - PyTorch implementation based on T5 architecture.

    This implementation is designed to be compatible with converted T5X checkpoints
    and follows the exact specifications for the MT3 model architecture.
    """

    def __init__(self, config: MT3Config):
        super().__init__()
        self.config = config

        # Shared embedding layer (T5 standard)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize encoder and decoder
        self.encoder = MT3Encoder(config, self.shared)
        self.decoder = MT3Decoder(config, self.shared)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.init_weights()

        # Tie weights (T5 standard - shared embeddings)
        self._tie_weights()

    def _tie_weights(self):
        """Tie shared embedding weights."""
        self.encoder.embed_tokens.weight = self.shared.weight
        self.decoder.embed_tokens.weight = self.shared.weight
        self.lm_head.weight = self.shared.weight

    def init_weights(self):
        """Initialize model weights."""
        # Initialize all weights with normal distribution
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                # T5 uses normal init with std=1.0 / sqrt(d_model)
                module.weight.data.normal_(mean=0.0, std=self.config.d_model ** -0.5)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.d_model ** -0.5)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

        self.apply(_init_weights)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = new_embeddings
        self.decoder.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tuple[Tensor]] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], Dict[str, Any]]:
        """
        Forward pass through the MT3 model.

        Args:
            input_ids: Encoder input token IDs [batch_size, seq_len]
            attention_mask: Encoder attention mask
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Decoder attention mask
            encoder_outputs: Cached encoder outputs
            past_key_values: Cached key/value states for generation
            use_cache: Whether to cache key/value states
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary

        Returns:
            Dict or tuple containing model outputs
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, 'use_cache') else True
        return_dict = return_dict if return_dict is not None else True

        # Encoder forward pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs['last_hidden_state'] if return_dict else encoder_outputs[0]

        # Decoder forward pass
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # For inference, start with decoder_start_token_id
            if input_ids is not None:
                batch_size = input_ids.shape[0]
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    self.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                raise ValueError("Must provide either decoder_input_ids or decoder_inputs_embeds")

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs['last_hidden_state'] if return_dict else decoder_outputs[0]

        # Language modeling head
        lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        return {
            'logits': lm_logits,
            'past_key_values': decoder_outputs.get('past_key_values'),
            'decoder_hidden_states': decoder_outputs.get('hidden_states'),
            'decoder_attentions': decoder_outputs.get('attentions'),
            'cross_attentions': decoder_outputs.get('cross_attentions'),
            'encoder_last_hidden_state': encoder_outputs.get('last_hidden_state'),
            'encoder_hidden_states': encoder_outputs.get('hidden_states'),
            'encoder_attentions': encoder_outputs.get('attentions'),
        }

    def generate(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        max_length: int = None,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        early_stopping: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        **kwargs
    ) -> Tensor:
        """
        Generate token sequences using the model.

        Args:
            input_ids: Input token IDs for encoder (for text/discrete inputs)
            inputs_embeds: Pre-embedded inputs for encoder (for audio/continuous inputs)
            encoder_outputs: Pre-computed encoder outputs
            attention_mask: Attention mask for encoder
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling
            early_stopping: Whether to stop early on EOS

        Returns:
            Generated token sequences [batch_size, seq_len]
        """
        max_length = max_length or self.config.max_length
        pad_token_id = pad_token_id or self.config.pad_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id

        if input_ids is None and inputs_embeds is None and encoder_outputs is None:
            raise ValueError("Must provide either input_ids, inputs_embeds, or encoder_outputs")

        # Determine batch size and device
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            batch_size = encoder_outputs['last_hidden_state'].shape[0]
            device = encoder_outputs['last_hidden_state'].device

        # Encode input if not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        # Initialize decoder input with decoder_start_token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        past_key_values = None

        for _ in range(max_length - 1):  # -1 because we start with one token
            # Forward pass
            outputs = self.forward(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids[:, -1:] if past_key_values is not None else decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs['logits'][:, -1, :]  # Last token logits
            past_key_values = outputs['past_key_values']

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_top_k, torch.full_like(logits, -float('inf')), logits)

            # Apply top-p filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('inf'))

            # Sample next tokens
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)

            # Check for EOS
            if early_stopping and (next_tokens == eos_token_id).all():
                break

        return decoder_input_ids

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_parameter_summary(self) -> Dict[str, int]:
        """Get detailed parameter count by component."""
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'total': self.get_parameter_count(),
            'shared_embeddings': count_parameters(self.shared),
            'encoder': count_parameters(self.encoder),
            'decoder': count_parameters(self.decoder),
            'lm_head': count_parameters(self.lm_head),
        }


def create_mt3_model(config: Optional[MT3Config] = None) -> MT3Model:
    """
    Create MT3Model with default or custom configuration.

    Args:
        config: Optional custom configuration

    Returns:
        Initialized MT3Model
    """
    if config is None:
        config = MT3Config()

    model = MT3Model(config)

    # Print parameter summary
    param_summary = model.get_parameter_summary()
    print(f"MT3Model created with {param_summary['total']:,} parameters")
    print(f"  - Shared embeddings: {param_summary['shared_embeddings']:,}")
    print(f"  - Encoder: {param_summary['encoder']:,}")
    print(f"  - Decoder: {param_summary['decoder']:,}")
    print(f"  - LM Head: {param_summary['lm_head']:,}")

    return model


# Example usage and validation
if __name__ == "__main__":
    # Create model with default configuration
    model = create_mt3_model()

    # Test forward pass
    batch_size = 2
    seq_len = 256
    vocab_size = 1536

    # Example inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, 32))

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

    print(f"Forward pass successful!")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Expected shape: [{batch_size}, 32, {vocab_size}]")

    # Test generation
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:1],  # Single example
            max_length=50,
            do_sample=False,
        )

    print(f"Generation successful!")
    print(f"Generated shape: {generated.shape}")