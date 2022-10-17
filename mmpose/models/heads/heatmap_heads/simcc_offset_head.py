# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


class GAU(nn.Module):

    def __init__(self,
                 max_seq_length,
                 hidden_size,
                 output_size,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5):

        super(GAU, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
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

        # self.log_n = math.log(max_seq_length)
        self.sqrt_s = math.sqrt(s)

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
            torch.arange(total_len, dtype=torch.float, device=x.device),
            spatial_shape)

        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)

        half_size = shape[-1] // 2
        freq_seq = -torch.arange(
            half_size, dtype=torch.float, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq
        sinusoid = torch.einsum('...,d->...d', position, inv_freq)
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
        else:
            # raise Exception("sequence length error.")
            a = self.rope(self.a.repeat(seq_len, 1), dim=0)
            b = self.rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.einsum('mk,nk->mn', a, b)
        return t

    def forward(self, x):
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
        base = torch.einsum('...r, hr->...hr', base, self.gamma) + self.beta
        base = self.rope(base, dim=1)
        q, k = torch.unbind(base, dim=-2)

        qk = torch.einsum('bnd,bmd->bnm', q, k)

        bias = self.rel_pos_bias(
            self.max_seq_length)[:, :seq_length, :seq_length]
        kernel = torch.square(F.relu(qk / self.sqrt_s + bias))
        # kernel = F.softmax(
        #  self.log_n * self.max_seq_length * qk / self.sqrt_s + bias, dim=-1)
        x = u * torch.einsum('bnm, bme->bne', kernel, v)
        x = self.o(x)
        if self.use_shortcut:
            x += shortcut
        return x


@MODELS.register_module()
class SimCC_Offset_Head(BaseHead):

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        debias: bool = False,
        beta: float = 1.,
        use_mlp: bool = False,
        softmax_norm: bool = False,
        input_transform: str = 'select',
        input_index: Union[int, Sequence[int]] = -1,
        align_corners: bool = False,
        simcc_loss: ConfigType = dict(
            type='KLDiscretLoss', use_target_weight=True),
        reg_loss: ConfigType = dict(type='RLELoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.align_corners = align_corners
        self.input_transform = input_transform
        self.input_index = input_index
        self.debias = debias
        self.beta = beta
        self.use_mlp = use_mlp
        self.softmax_norm = softmax_norm
        self.reg_loss = MODELS.build(reg_loss)
        self.simcc_loss = MODELS.build(simcc_loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, list):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        in_channels = self._get_in_channels()

        cfg = dict(
            type='Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)
        self.final_layer = build_conv_layer(cfg)
        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = GAU(self.out_channels, flatten_dims, W)
        self.mlp_head_y = GAU(self.out_channels, flatten_dims, H)

        # Define rle
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.rle_head = nn.Linear(in_channels, out_channels * 4)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = self._transform_inputs(feats)
        B, C = feats.shape[:2]

        output_coord = self.rle_head(self.gap(feats).reshape(B, C))  # B, K*4
        output_coord = output_coord.reshape(B, -1, 4)

        feats = self.final_layer(feats)

        # flatten the output heatmap
        x = torch.flatten(feats, 2)

        simcc_x = self.mlp_head_x(x)
        simcc_y = self.mlp_head_y(x)

        return output_coord, simcc_x, simcc_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']

            _feats, _feats_flip = feats

            _batch_coords, _, _ = self.forward(_feats)
            _batch_coords[..., 2:] = _batch_coords[..., 2:].sigmoid()

            _batch_coords_flip = flip_coordinates(
                self.forward(_feats_flip)[0],
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            _batch_coords_flip[..., 2:] = _batch_coords_flip[..., 2:].sigmoid()

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords, _, _ = self.forward(feats)  # (B, K, D)
            batch_coords[..., 2:] = batch_coords[..., 2:].sigmoid()

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        return preds

    def loss(
        self,
        inputs: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_outputs, pred_x, pred_y = self.forward(inputs)

        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        pred_coords = pred_outputs[:, :, :2]
        pred_sigma = pred_outputs[:, :, 2:4]

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights2 = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.reg_loss(pred_coords, pred_sigma, keypoint_labels,
                             keypoint_weights)
        loss += self.simcc_loss(pred_simcc, gt_simcc, keypoint_weights2)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_coords),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
