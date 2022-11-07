# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath


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


class PGAU(nn.Module):

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
            coord_dims=512,
            act_fn='SiLU',
            bias=False,
            use_rel_bias=True):

        super(PGAU, self).__init__()
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
                self.a = nn.Parameter(
                    torch.rand([1, coord_dims], dtype=torch.float))
                self.b = nn.Parameter(
                    torch.rand([1, coord_dims], dtype=torch.float))
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

        self.layer_scale = Scale(in_token_dims)

        self.sqrt_s = math.sqrt(s)

        self.dropout_rate = dropout_rate

        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

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
            half_size, dtype=torch.int, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq

        sinusoid = position[..., None] * inv_freq[None, None, :]

        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)

        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rel_pos_bias(self, seq_len):
        if self.shift == 'structure':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = self.rope(self.a.repeat(seq_len, 1), dim=0)
            b = self.rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def shift_mixing(self, x, shift_type='time'):
        x_shift, x_pass = x.chunk(2, dim=-1)
        if shift_type == 'structure' and len(self.shift_idx) == x.size(1):
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

            if self.use_rel_bias:
                base = self.rope(base, dim=1)

            q, k = torch.unbind(base, dim=-2)

        else:
            u, q = torch.split(self.act_fn(uv), [self.e, self.s], dim=-1)

            if self.shift is not None:
                k = self.shift_mixing(k)
                v = self.shift_mixing(v)

            k = self.k_fc(k)
            v = self.v_fc(v)

            if self.use_rel_bias:
                q = self.rope(q, 1)
                k = self.rope(k, 1)

        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            bias = self.rel_pos_bias(max(q.size(1),
                                         k.size(1)))[:, :q.size(1), :k.size(1)]
            qk += bias

        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate:
            kernel = self.dropout(kernel)

        x = u * torch.bmm(kernel, v)
        x = self.o(x)

        return x

    def forward(self, x):
        if self.shortcut:
            return self.res_scale(x) + \
                self.layer_scale(self.drop_path(self._forward(x)))
        else:
            return self.layer_scale(self.drop_path(self._forward(x)))


class GAU(nn.Module):

    def __init__(self,
                 max_seq_length,
                 hidden_size,
                 output_size,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 use_dropout=False,
                 drop_path=0.,
                 attn='relu2',
                 kpt_structure=False,
                 self_attn=True,
                 shift=False,
                 coord_dims=512,
                 act='silu',
                 bias=False,
                 use_rel_bias=True):

        super(GAU, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
        self.attn = attn
        self.shift = shift
        self.drop_path = drop_path
        self.use_rel_bias = use_rel_bias

        self.e = int(hidden_size * expansion_factor)
        if use_rel_bias:
            if self_attn:
                self.w = nn.Parameter(
                    torch.rand([2 * max_seq_length - 1], dtype=torch.float))
            else:
                self.w = nn.Parameter(
                    torch.rand([2 * coord_dims - 1], dtype=torch.float))
        # self.a = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        # self.b = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        self.o = nn.Linear(self.e, output_size, bias=bias)

        if self_attn:
            self.uv = nn.Linear(hidden_size, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(hidden_size, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(hidden_size, self.s, bias=bias)
            self.v_fc = nn.Linear(hidden_size, self.e, bias=bias)

        # self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.ln = ScaleNorm(hidden_size, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)

        if act == 'silu':
            self.act_fn = nn.SiLU(True)
        elif act == 'starrelu':
            self.act_fn = StarReLU(True)
        else:
            self.act_fn = nn.ReLU6(True)

        self.use_shortcut = hidden_size == output_size

        if attn == 'softmax':
            self.log_n = math.log(max_seq_length)
        elif attn == 'laplacian':
            self.mu = math.sqrt(0.5)
            self.std = math.sqrt(0.25 * math.pi)
        elif attn == 'starrelu':
            self.star_relu = StarReLU()

        self.sqrt_s = math.sqrt(s)

        self.use_dropout = use_dropout

        if use_dropout:
            self.dropout = nn.Dropout(0.3)

        self.kpt_structure = kpt_structure

        self.self_attn = self_attn

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

        if self.kpt_structure and spatial_shape[0] == 17:
            position = torch.tensor(
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                dtype=torch.float,
                device=x.device).reshape(spatial_shape)
        else:
            position = torch.reshape(
                torch.arange(total_len, dtype=torch.int, device=x.device),
                spatial_shape)

        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)

        half_size = shape[-1] // 2
        freq_seq = -torch.arange(
            half_size, dtype=torch.int, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq
        # sinusoid = torch.einsum('...,d->...d', position, inv_freq)
        sinusoid = position[..., None] * inv_freq[None, None, :]
        # print(torch.sum(sinusoid -sinusoid2))
        # print(position.shape, inv_freq.shape, sinusoid.shape)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rel_pos_bias(self, seq_len):

        t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        t = t[..., r:-r]
        # else:
        # #     # raise Exception("sequence length error.")
        # a = self.rope(self.a.repeat(seq_len, 1), dim=0)
        # b = self.rope(self.b.repeat(seq_len, 1), dim=0)
        # t = torch.einsum('bmk,bnk->bmn', a, b)
        # t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def forward(self, inputs):
        """
        :param x:  [batch_size, sequence_length, model_dim]
        :param causal:add mask tensor matrix
        :return:
        """
        if self.self_attn:
            x = inputs
        else:
            x, k, v = inputs

        # seq_length = x.shape[1]
        if self.use_shortcut:
            shortcut = x

        x = self.ln(x)

        if self.shift:
            x_shift, x_pass = x.chunk(2, dim=-1)
            if self.max_seq_length != 17:
                x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            else:
                shift_idx = [
                    0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14
                ]
                x_shift = x_shift[:, shift_idx, :]
            x = torch.cat((x_shift, x_pass), dim=-1)

        uv = self.uv(x)
        if self.self_attn:
            u, v, base = torch.split(
                self.act_fn(uv), [self.e, self.e, self.s], dim=-1)

            # base1 = torch.einsum('...r, hr->...hr', base, self.gamma)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            base = self.rope(base, dim=1)
            q, k = torch.unbind(base, dim=-2)

        else:
            u, q = torch.split(self.act_fn(uv), [self.e, self.s], dim=-1)

            if self.shift:
                k_shift, k_pass = k.chunk(2, dim=-1)
                k_shift = F.pad(k_shift, (0, 0, 1, -1), value=0.)
                k = torch.cat((k_shift, k_pass), dim=-1)

                v_shift, v_pass = v.chunk(2, dim=-1)
                v_shift = F.pad(v_shift, (0, 0, 1, -1), value=0.)
                v = torch.cat((v_shift, v_pass), dim=-1)
            k = self.k_fc(k)
            v = self.v_fc(v)
            q = self.rope(q, 1)
            k = self.rope(k, 1)

        # qk = torch.einsum('bnd,bmd->bnm', q, k)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        # print('q', q.shape, 'k', k.shape, 'qk', qk.shape)
        if self.use_rel_bias:
            bias = self.rel_pos_bias(max(q.size(1),
                                         k.size(1)))[:, :q.size(1), :k.size(1)]
            # print('bias', bias.shape)
            qk += bias

        if self.attn == 'softmax':
            kernel = F.softmax(
                self.log_n * self.max_seq_length * qk / self.sqrt_s, dim=-1)
        elif self.attn == 'laplacian':
            kernel = (1 + torch.special.erf(
                (qk - self.mu) / (self.std * math.sqrt(2)))) * 0.5 / self.s
        elif self.attn == 'starrelu':
            kernel = self.star_relu(qk / self.sqrt_s)
        else:
            kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.use_dropout:
            kernel = self.dropout(kernel)
        # print(kernel.shape)
        # x = u * torch.einsum('bnm, bme->bne', kernel, v)
        x = u * torch.bmm(kernel, v)
        # print(torch.sum(x-x2))

        x = self.o(x)

        if self.use_shortcut:
            if self.training and self.drop_path > np.random.rand():
                x = shortcut
            else:
                x += shortcut
        return x


class GAUAlpha(nn.Module):

    def __init__(self,
                 max_seq_length,
                 hidden_size,
                 output_size,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 use_dropout=False,
                 attn='relu2',
                 kpt_structure=False,
                 self_attn=True,
                 shift=False,
                 coord_dims=512):

        super(GAUAlpha, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
        self.attn = attn
        self.shift = shift

        self.e = int(hidden_size * expansion_factor)
        if self_attn:
            self.w = nn.Parameter(
                torch.rand([2 * max_seq_length - 1], dtype=torch.float))
        else:
            self.w = nn.Parameter(
                torch.rand([2 * coord_dims - 1], dtype=torch.float))
        # self.a = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        # self.b = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        self.o = nn.Linear(self.e, output_size)

        if self_attn:
            self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(hidden_size, self.e + self.s)
            self.k_fc = nn.Linear(hidden_size, self.s)
            self.v_fc = nn.Linear(hidden_size, self.e)
            self.o2 = nn.Linear(self.e, output_size)

        # self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.ln = ScaleNorm(hidden_size, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)

        self.act_fn = nn.SiLU(True)

        self.use_shortcut = hidden_size == output_size

        if attn == 'softmax':
            self.log_n = math.log(max_seq_length)
        elif attn == 'laplacian':
            self.mu = math.sqrt(0.5)
            self.std = math.sqrt(0.25 * math.pi)
        elif attn == 'starrelu':
            self.star_relu = StarReLU()

        self.sqrt_s = math.sqrt(s)

        self.use_dropout = use_dropout

        if use_dropout:
            self.dropout = nn.Dropout(0.3)

        self.kpt_structure = kpt_structure

        self.self_attn = self_attn

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

        if self.kpt_structure and spatial_shape[0] == 17:
            position = torch.tensor(
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                dtype=torch.float,
                device=x.device).reshape(spatial_shape)
        else:
            position = torch.reshape(
                torch.arange(total_len, dtype=torch.int, device=x.device),
                spatial_shape)

        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)

        half_size = shape[-1] // 2
        freq_seq = -torch.arange(
            half_size, dtype=torch.int, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq
        # sinusoid = torch.einsum('...,d->...d', position, inv_freq)
        sinusoid = position[..., None] * inv_freq[None, None, :]
        # print(torch.sum(sinusoid -sinusoid2))
        # print(position.shape, inv_freq.shape, sinusoid.shape)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rel_pos_bias(self, seq_len):
        if seq_len <= 512:
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        # else:
        # #     # raise Exception("sequence length error.")
        # a = self.rope(self.a.repeat(seq_len, 1), dim=0)
        # b = self.rope(self.b.repeat(seq_len, 1), dim=0)
        # t = torch.einsum('bmk,bnk->bmn', a, b)
        # t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def forward(self, inputs):
        """
        :param x:  [batch_size, sequence_length, model_dim]
        :param causal:add mask tensor matrix
        :return:
        """
        if self.self_attn:
            x = inputs
        else:
            x, k, v = inputs
            if self.use_shortcut:
                v_shortcut = v

        # seq_length = x.shape[1]
        if self.use_shortcut:
            shortcut = x

        x = self.ln(x)

        if self.shift:
            x_shift, x_pass = x.chunk(2, dim=-1)
            if self.max_seq_length != 17:
                x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            else:
                shift_idx = [
                    0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14
                ]
                x_shift = x_shift[:, shift_idx, :]
            x = torch.cat((x_shift, x_pass), dim=-1)

        uv = self.uv(x)
        if self.self_attn:
            u, v, base = torch.split(
                self.act_fn(uv), [self.e, self.e, self.s], dim=-1)

            # base1 = torch.einsum('...r, hr->...hr', base, self.gamma)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            base = self.rope(base, dim=1)
            q, k = torch.unbind(base, dim=-2)

        else:
            u, q = torch.split(self.act_fn(uv), [self.e, self.s], dim=-1)

            if self.shift:
                k_shift, k_pass = k.chunk(2, dim=-1)
                k_shift = F.pad(k_shift, (0, 0, 1, -1), value=0.)
                k = torch.cat((k_shift, k_pass), dim=-1)

                v_shift, v_pass = v.chunk(2, dim=-1)
                v_shift = F.pad(v_shift, (0, 0, 1, -1), value=0.)
                v = torch.cat((v_shift, v_pass), dim=-1)
            k = self.k_fc(k)
            v = self.v_fc(v)
            q = self.rope(q, 1)
            k = self.rope(k, 1)

        # qk = torch.einsum('bnd,bmd->bnm', q, k)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        # print('q', q.shape, 'k', k.shape, 'qk', qk.shape)
        bias = self.rel_pos_bias(max(q.size(1),
                                     k.size(1)))[:, :q.size(1), :k.size(1)]
        # print('bias', bias.shape)
        qk += bias

        if self.attn == 'softmax':
            kernel = F.softmax(
                self.log_n * self.max_seq_length * qk / self.sqrt_s, dim=-1)
        elif self.attn == 'laplacian':
            kernel = (1 + torch.special.erf(
                (qk - self.mu) / (self.std * math.sqrt(2)))) * 0.5 / self.s
        elif self.attn == 'starrelu':
            kernel = self.star_relu(qk / self.sqrt_s)
        else:
            kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.use_dropout:
            kernel = self.dropout(kernel)
        # print(kernel.shape)
        # x = u * torch.einsum('bnm, bme->bne', kernel, v)
        x = u * torch.bmm(kernel, v)

        x = self.o(x)

        if self.use_shortcut:
            x += shortcut

        if self.self_attn is False:
            # cross attn
            # kernel B, 17, 512
            # v  B, 512, e
            # u  B, 17, e
            y = v * torch.bmm(kernel.permute(0, 2, 1), u)  # B, 512, e
            y = self.o2(y)

            if self.use_shortcut:
                y += v_shortcut

            return x, y
        return x


class SAGAU(nn.Module):

    def __init__(self,
                 max_seq_length,
                 hidden_size,
                 output_size,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 use_dropout=False,
                 softmax_att=False):

        super(SAGAU, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
        self.softmax_att = softmax_att
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.e = int(hidden_size * expansion_factor)
        self.w = nn.Parameter(
            torch.rand([2 * max_seq_length - 1], dtype=torch.float))
        # self.a = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        # self.b = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        self.o = nn.Linear(self.e, output_size)
        self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)
        self.act_fn = nn.SiLU(True)
        self.use_shortcut = hidden_size == output_size
        if softmax_att:
            self.log_n = math.log(max_seq_length)
        self.sqrt_s = math.sqrt(s)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.2)

    def rope(self, x, proposal=None, dim=1):
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

        if proposal is None:
            position = torch.reshape(
                torch.arange(total_len, dtype=torch.int, device=x.device),
                spatial_shape)
        else:
            position = proposal

        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)

        half_size = shape[-1] // 2
        freq_seq = -torch.arange(
            half_size, dtype=torch.int, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq
        # sinusoid = torch.einsum('...,d->...d', position, inv_freq)
        if proposal is None:
            sinusoid = position[..., None] * inv_freq[None, None, :]
        else:
            sinusoid = position * inv_freq[None, None, :]
        # print(torch.sum(sinusoid -sinusoid2))
        # print(position.shape, inv_freq.shape, sinusoid.shape)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rel_pos_bias(self, seq_len):
        if seq_len <= 512:
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        # else:
        #     # raise Exception("sequence length error.")
        #     a = self.rope(self.a.repeat(seq_len, 1), dim=0)
        #     b = self.rope(self.b.repeat(seq_len, 1), dim=0)
        #     t = torch.einsum('mk,nk->mn', a, b)
        return t

    def _get_proposal_pos_embed(self,
                                proposals,
                                num_pos_feats=128,
                                temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.int, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        proposals = proposals * scale

        # N, L, 2, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self, x, proposal=None):
        """
        :param x:  [batch_size, sequence_length, model_dim]
        :param causal:add mask tensor matrix
        :return:
        """
        seq_length = x.shape[1]
        if self.use_shortcut:
            shortcut = x
        x = self.ln(x)
        uv = self.uv(x)
        u, v, base = torch.split(
            self.act_fn(uv), [self.e, self.e, self.s], dim=-1)
        # print(base.shape, self.gamma.shape)
        # base1 = torch.einsum('...r, hr->...hr', base, self.gamma)
        base = base.unsqueeze(2) * self.gamma[None, None, :]
        # print(torch.sum(base1-base2))
        base = base + self.beta
        base = self.rope(base, proposal, dim=1)
        q, k = torch.unbind(base, dim=-2)

        # qk = torch.einsum('bnd,bmd->bnm', q, k)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        # print(torch.sum(qk-qk2))

        bias = self.rel_pos_bias(
            self.max_seq_length)[:, :seq_length, :seq_length]

        if self.softmax_att:
            kernel = F.softmax(
                self.log_n * self.max_seq_length * qk / self.sqrt_s + bias,
                dim=-1)
        else:
            kernel = torch.square(F.relu(qk / self.sqrt_s + bias))

        if self.use_dropout:
            kernel = self.dropout(kernel)

        # x = u * torch.einsum('bnm, bme->bne', kernel, v)
        x = u * torch.bmm(kernel, v)
        # print(torch.sum(x-x2))

        x = self.o(x)

        if self.use_shortcut:
            x += shortcut
        return x


class KeypointCoordMatching(nn.Module):

    def __init__(self, num_kpt, num_coord, hidden_dims=256, use_dropout=False):
        super(KeypointCoordMatching, self).__init__()
        self.kpt_gau_encoder = GAU(
            num_kpt, hidden_dims, hidden_dims, use_dropout=use_dropout)
        self.coord_gau_encoder = GAU(
            num_coord, hidden_dims, hidden_dims, use_dropout=use_dropout)
        self.kpt_gau_decoder = GAU(
            num_kpt,
            hidden_dims,
            hidden_dims,
            self_attn=False,
            use_dropout=use_dropout)
        self.coord_gau_decoder = GAU(
            num_coord,
            hidden_dims,
            hidden_dims,
            self_attn=False,
            use_dropout=use_dropout)

    def forward(self, kpt_token, coord_token):
        kpt_token = self.kpt_gau_encoder(kpt_token)
        coord_token = self.coord_gau_encoder(coord_token)
        kpt_token = self.kpt_gau_decoder((kpt_token, coord_token, coord_token))
        coord_token = self.coord_gau_decoder(
            (coord_token, kpt_token, kpt_token))
        return kpt_token, coord_token


class KCM(nn.Module):

    def __init__(self,
                 kpt_dim,
                 coord_dim,
                 num_kpt_enc,
                 num_coord_enc,
                 num_kpt_dec,
                 num_coord_dec,
                 k2c=True,
                 c2k=True,
                 hidden_dims=256,
                 s=128,
                 use_dropout=False,
                 shift=False,
                 attn='relu2'):
        super(KCM, self).__init__()
        self.k2c = k2c
        self.c2k = c2k
        self.num_kpt_dec = num_kpt_dec

        kpt_encoders = [
            GAU(kpt_dim,
                hidden_dims,
                hidden_dims,
                attn=attn,
                shift=shift,
                use_dropout=use_dropout,
                s=s) for _ in range(num_kpt_enc)
        ]
        coord_encoders = [
            GAU(coord_dim,
                hidden_dims,
                hidden_dims,
                attn=attn,
                shift=shift,
                use_dropout=use_dropout,
                s=s) for _ in range(num_coord_enc)
        ]
        self.kpt_encoder = nn.Sequential(*kpt_encoders)
        self.coord_encoder = nn.Sequential(*coord_encoders)

        if self.c2k:
            kpt_decoders = [
                GAU(kpt_dim,
                    hidden_dims,
                    hidden_dims,
                    self_attn=False,
                    attn=attn,
                    shift=shift,
                    use_dropout=use_dropout,
                    s=s) for _ in range(num_kpt_dec)
            ]
            self.kpt_decoder = nn.ModuleList(kpt_decoders)

        if self.k2c:
            coord_decoders = [
                GAU(coord_dim,
                    hidden_dims,
                    hidden_dims,
                    self_attn=False,
                    attn=attn,
                    shift=shift,
                    use_dropout=use_dropout,
                    s=s) for _ in range(num_coord_dec)
            ]
            self.coord_decoder = nn.ModuleList(coord_decoders)

    def forward(self, kpt_token, coord_token):
        kpt_token = self.kpt_encoder(kpt_token)
        coord_token = self.coord_encoder(coord_token)

        for i in range(self.num_kpt_dec):
            if self.k2c:
                coord_token = self.coord_decoder[i](
                    (coord_token, kpt_token, kpt_token))
            if self.c2k:
                kpt_token = self.kpt_decoder[i](
                    (kpt_token, coord_token, coord_token))
        return kpt_token, coord_token


if __name__ == '__main__':
    q = torch.rand(4, 17, 256)
    m = torch.rand(4, 512, 256)
    # gau = GAU(17, 256, 256, attn='laplacian', shift=True)
    kc = KCM(17, 256, 1, 1, 1, 1, shift=True, attn='laplacian')
    # p = torch.rand(4, 17, 1)
    # kc = KeypointCoordMatching(17, 512)
    # res = gau(q)
    # print(res.shape)
    # res = gau((q, m, m))
    res1, res2 = kc(q, m)
    print(res1.shape, res2.shape)
