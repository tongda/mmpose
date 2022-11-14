# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath


def rope(x, dim):
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
        half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000**-freq_seq

    sinusoid = position[..., None] * inv_freq[None, None, :]

    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SE(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        w = self.block(x).sigmoid()
        x = x * w
        return x


class Scale(nn.Module):
    """Scale vector by element multiplications."""

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class ScaleNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 scale_learnable=True,
                 bias_learnable=True,
                 mode=None,
                 inplace=False):

        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(
            scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(
            bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class RTMBlock(nn.Module):

    def __init__(
            self,
            num_token,
            in_token_dims,
            out_token_dims,
            expansion_factor=2,
            s=128,
            eps=1e-5,
            dropout_rate=0.,
            drop_path=0.,
            attn_type='self-attn',
            shift=None,
            shift_idx=[0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14],
            act_fn='SiLU',
            bias=False,
            use_rel_bias=True):

        super(RTMBlock, self).__init__()
        self.s = s
        self.num_token = num_token
        self.shift = shift
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.shift_idx = shift_idx
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(
                    torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == 'SiLU':
            self.act_fn = nn.SiLU(True)
        elif act_fn == 'StarReLU':
            self.act_fn = StarReLU(True)
        else:
            self.act_fn = nn.ReLU(True)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        # self.layer_scale = Scale(out_token_dims)

        self.sqrt_s = math.sqrt(s)

        self.dropout_rate = dropout_rate

        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def shift_mixing(self, x):
        x_shift, x_pass = x.chunk(2, dim=-1)
        if self.shift == 'structure' and len(self.shift_idx) == x.size(1):
            x_shift = x_shift[:, self.shift_idx, :]
        else:
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)

        x = torch.cat((x_shift, x_pass), dim=-1)
        return x

    def _forward(self, inputs):
        """
        :param x:  [batch_size, sequence_length, model_dim]
        :param causal:add mask tensor matrix
        :return:
        """
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)

        if self.shift is not None:
            x = self.shift_mixing(x)

        uv = self.uv(x)

        if self.attn_type == 'self-attn':
            u, v, base = torch.split(
                self.act_fn(uv), [self.e, self.e, self.s], dim=-1)

            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            base = rope(base, dim=1)

            q, k = torch.unbind(base, dim=-2)

        else:
            u, q = torch.split(self.act_fn(uv), [self.e, self.s], dim=-1)

            if self.shift is not None:
                k = self.shift_mixing(k)
                v = self.shift_mixing(v)

            k = self.k_fc(k)
            v = self.v_fc(v)

            q = rope(q, 1)
            k = rope(k, 1)

        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]

        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)

        x = u * torch.bmm(kernel, v)
        x = self.o(x)

        return x

    def forward(self, x):
        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            # return self.res_scale(res_shortcut)
            # + self.layer_scale(main_branch)
            return self.res_scale(res_shortcut) + main_branch
            # return res_shortcut + main_branch
        else:
            # return self.layer_scale(self.drop_path(self._forward(x)))
            return self.drop_path(self._forward(x))
