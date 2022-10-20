# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class ScaleNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class T5RelativePositionBias(nn.Module):

    def __init__(self, scale, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) /
                                    math.log(max_distance / max_exact) *
                                    (num_buckets - max_exact)).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale


class OffsetScale(nn.Module):

    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class ReLUSquared(nn.Module):

    def forward(self, x):
        return F.relu(x)**2


class LaplacianAttnFn(nn.Module):
    """https://arxiv.org/abs/2209.10655 claims this is more stable than Relu
    squared."""

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt(0.25 * math.pi)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5


class FLASH(nn.Module):

    def __init__(self,
                 *,
                 dim,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=2.,
                 dropout=0.,
                 rotary_pos_emb=None,
                 norm_klass=nn.LayerNorm,
                 shift_tokens=False,
                 laplace_attn_fn=False,
                 reduce_group_non_causal_attn=True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.shift_tokens = shift_tokens

        self.attn_fn = ReLUSquared(
        ) if not laplace_attn_fn else LaplacianAttnFn()

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = None

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2), nn.SiLU())

        self.to_qk = nn.Sequential(nn.Linear(dim, query_key_dim), nn.SiLU())

        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = nn.Linear(hidden_dim, dim)

    def rope(self, x, dim):
        """
        :param x: input tensor
        :param dim: operation dimension
        :return:
        """
        shape = x.shape
        if isinstance(dim, int):
            dim = [dim]

        spatial_shape = [shape[i] for i in dim]
        total_len = 1
        for i in spatial_shape:
            total_len *= i

        position = torch.reshape(
            torch.arange(total_len, dtype=torch.int, device=x.device),
            spatial_shape)

        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)

        half_size = shape[-1] // 2
        freq_seq = -torch.arange(
            half_size, dtype=torch.float, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq
        # sinusoid = torch.einsum('...,d->...d', position, inv_freq)
        sinusoid = position[..., None] * inv_freq[None, None, :]
        # print(torch.sum(sinusoid -sinusoid2))
        # print(position.shape, inv_freq.shape, sinusoid.shape)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        x,
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        n, g = x.shape[-2], self.group_size

        # prenorm

        normed_x = self.norm(x)

        # do token shift - a great, costless trick from an independent
        # AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # rotate queries and keys
        quad_q, lin_q, quad_k, lin_k = map(self.rope,
                                           (quad_q, lin_q, quad_k, lin_k))

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.),
                (quad_q, quad_k, lin_q, lin_k, v))

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(
            lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v))

        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        sim = sim + self.rel_pos_bias(sim)

        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output
        if self.reduce_group_non_causal_attn:
            context_einsum_eq = 'b d e'
        else:
            context_einsum_eq = 'b g d e'
        lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k,
                        v) / n
        lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q,
                         lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(
            lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n],
            (quad_out, lin_out))

        # gate

        out = gate * (quad_attn_out + lin_attn_out)

        # projection out and residual

        return self.to_out(out) + x
