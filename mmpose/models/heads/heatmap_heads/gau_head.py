# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from torch import Tensor, nn

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.tta import flip_vectors
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
class GAU_Head(BaseHead):
    """Top-down heatmap head introduced in `SimCC`_ by Li et al (2022). The
    head is composed of a few deconvolutional layers followed by a fully-
    connected layer to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        input_size (tuple): Input image size in shape [w, h]
        in_featuremap_size (int | sequence[int]): Size of input feature map
        simcc_split_ratio (float): Split ratio of pixels
        deconv_type (str, optional): The type of deconv head which should
            be one of the following options:

                - ``'heatmap'``: make deconv layers in `HeatmapHead`
                - ``'vipnas'``: make deconv layers in `ViPNASHead`

            Defaults to ``'Heatmap'``
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        deconv_num_groups (Sequence[int], optional): The group number of each
            deconv layer. Defaults to ``(16, 16, 16)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        input_transform (str): Transformation of input features which should
            be one of the following options:

                - ``'resize_concat'``: Resize multiple feature maps specified
                    by ``input_index`` to the same size as the first one and
                    concat these feature maps
                - ``'select'``: Select feature map(s) specified by
                    ``input_index``. Multiple selected features will be
                    bundled into a tuple

            Defaults to ``'select'``
        input_index (int | sequence[int]): The feature map index used in the
            input transformation. See also ``input_transform``. Defaults to -1
        align_corners (bool): `align_corners` argument of
            :func:`torch.nn.functional.interpolate` used in the input
            transformation. Defaults to ``False``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`SimCC`: https://arxiv.org/abs/2107.03332
    """

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        hidden_dims: int = 256,
        num_global: int = 1,
        num_split: int = 1,
        deconv_type: str = 'heatmap',
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        has_final_layer: bool = True,
        input_transform: str = 'select',
        input_index: Union[int, Sequence[int]] = -1,
        align_corners: bool = False,
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        if deconv_type not in {'heatmap', 'vipnas'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `deconv_type` value'
                f'{deconv_type}. Should be one of '
                '{"heatmap", "vipnas"}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.num_global = num_global
        self.num_split = num_split
        self.align_corners = align_corners
        self.input_transform = input_transform
        self.input_index = input_index
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple(
                [s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.deconv_head = self._make_deconv_head(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv_type=deconv_type,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                deconv_num_groups=deconv_num_groups,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                has_final_layer=has_final_layer,
                input_transform=input_transform,
                input_index=input_index,
                align_corners=align_corners)

            if has_final_layer:
                in_channels = out_channels
            else:
                in_channels = deconv_out_channels[-1]

        else:
            in_channels = self._get_in_channels()
            self.deconv_head = None

            if has_final_layer:
                cfg = dict(
                    type='Conv2d',
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1)
                self.final_layer = build_conv_layer(cfg)
            else:
                self.final_layer = None

            if self.input_transform == 'resize_concat':
                if isinstance(in_featuremap_size, tuple):
                    self.heatmap_size = in_featuremap_size
                elif isinstance(in_featuremap_size, list):
                    self.heatmap_size = in_featuremap_size[0]
            elif self.input_transform == 'select':
                if isinstance(in_featuremap_size, tuple):
                    self.heatmap_size = in_featuremap_size
                elif isinstance(in_featuremap_size, list):
                    self.heatmap_size = in_featuremap_size[input_index]

        if isinstance(in_channels, list):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        global_gau = [GAU(self.out_channels, flatten_dims, hidden_dims)]
        for _ in range(num_global - 1):
            global_gau.append(GAU(self.out_channels, hidden_dims, hidden_dims))
        self.global_gau = nn.Sequential(*global_gau)

        gau_x, gau_y = [], []
        for _ in range(num_split):
            gau_x.append(GAU(self.out_channels, hidden_dims, hidden_dims))
            gau_y.append(GAU(self.out_channels, hidden_dims, hidden_dims))
        self.gau_x = nn.Sequential(*gau_x)
        self.gau_y = nn.Sequential(*gau_y)

        self.mlp_head_x = nn.Linear(hidden_dims, W)
        self.mlp_head_y = nn.Linear(hidden_dims, H)

    def _make_deconv_head(self,
                          in_channels: Union[int, Sequence[int]],
                          out_channels: int,
                          deconv_type: str = 'heatmap',
                          deconv_out_channels: OptIntSeq = (256, 256, 256),
                          deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                          deconv_num_groups: OptIntSeq = (16, 16, 16),
                          conv_out_channels: OptIntSeq = None,
                          conv_kernel_sizes: OptIntSeq = None,
                          has_final_layer: bool = True,
                          input_transform: str = 'select',
                          input_index: Union[int, Sequence[int]] = -1,
                          align_corners: bool = False) -> nn.Module:

        if deconv_type == 'heatmap':
            deconv_head = MODELS.build(
                dict(
                    type='HeatmapHead',
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_kernel_sizes=deconv_kernel_sizes,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    has_final_layer=has_final_layer,
                    input_transform=input_transform,
                    input_index=input_index,
                    align_corners=align_corners))
        else:
            deconv_head = MODELS.build(
                dict(
                    type='ViPNASHead',
                    in_channels=in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_num_groups=deconv_num_groups,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    has_final_layer=has_final_layer,
                    input_transform=input_transform,
                    input_index=input_index,
                    align_corners=align_corners))

        return deconv_head

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        if self.deconv_head is None:
            feats = self._transform_inputs(feats)
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.deconv_head(feats)

        # flatten the output heatmap
        x = torch.flatten(feats, 2)

        x = self.global_gau(x)

        pred_x = self.gau_x(x)
        pred_y = self.gau_y(x)

        pred_x = self.mlp_head_x(pred_x)
        pred_y = self.mlp_head_y(pred_y)

        return pred_x, pred_y

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
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):

                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
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
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
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
