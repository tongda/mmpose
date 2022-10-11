# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmengine.logging import MessageHub
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class SimCC_SamplingArgmax_Head(BaseHead):

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        num_sample: int = 1,
        basis_type: str = 'tri',
        debias: bool = False,
        beta: float = 1.,
        input_transform: str = 'select',
        input_index: Union[int, Sequence[int]] = -1,
        align_corners: bool = False,
        loss: ConfigType = dict(type='RLELoss', use_target_weight=True),
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
        self.basis_type = basis_type
        self.num_sample = num_sample
        self._tau = 2
        self.debias = debias
        self.beta = beta
        self.loss_module = MODELS.build(loss)
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

        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)

        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, W)
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H)

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

        # Define rle
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigma_head = nn.Linear(in_channels, out_channels * 2)

    def _normalize(self, feats, num_sample=1, tau=2):

        if self.training:
            B, C, Wx = feats.shape
            feats = feats.reshape(B, C, 1, Wx)

            eps = torch.rand(B, C, num_sample, Wx, device=feats.device)
            log_eps = torch.log(-torch.log(eps))
            gumbel_feats = feats - log_eps / tau
            gumbel_feats = F.softmax(gumbel_feats, dim=3)

            # hard_gumbel_feats = gumbel_feats.detach()
            # hard_gumbel_feats = (torch.max(
            #     hard_gumbel_feats, dim=-1,
            #     keepdim=True)[0] == hard_gumbel_feats).float()

            # gumbel_feats = (hard_gumbel_feats -
            #                 gumbel_feats).detach() + gumbel_feats

            return gumbel_feats
        else:
            feats = F.softmax(feats, dim=2)
            return feats

    def _uni2tri(self, eps):
        # eps U[0, 1]
        # PDF:
        # y = x + 1 (-1 < x < 0)
        # y = -x + 1 (0 < x < 1)
        # CDF:
        # y = x^2 / 2 + x + 1/2 (-1 < x < 0)
        # y = -x^2 / 2 + x + 1/2 (0 < x < 1)
        # invcdf:
        # x = sqrt(2y) - 1, y < 0.5
        # x = 1 - sqrt(2 - 2y), y > 0.5
        tri = torch.where(eps < 0.5,
                          torch.sqrt(2 * eps) - 1, 1 - torch.sqrt(2 - 2 * eps))
        p = torch.where(tri < 0, tri + 1, -tri + 1)
        return tri, p

    def _retrive_p(self, hm, x):
        # hm: (B, K, W) or (B, K, S, W)
        # x:  (B, K, W) or (B, K, S, W)
        left_x = x.floor() + 1
        right_x = (x + 1).floor() + 1
        left_hm = F.pad(hm, (1, 1)).gather(-1, left_x.long())
        right_hm = F.pad(hm, (1, 1)).gather(-1, right_x.long())
        new_hm = left_hm + (right_hm - left_hm) * (x + 1 - left_x)

        return new_hm

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

        output_sigma = self.sigma_head(self.gap(feats).reshape(B, C))  # B, K*2
        output_sigma = output_sigma.reshape(B, -1, 2)

        feats = self.final_layer(feats)

        # flatten the output heatmap
        x = torch.flatten(feats, 2)

        simcc_x = self.mlp_head_x(x)
        simcc_y = self.mlp_head_y(x)

        if self.training:
            mh = MessageHub.get_current_instance()
            idx_epoch = mh.get_info('epoch')
            self._tau = max(0.5, -idx_epoch * 3 / 50 + 2)

        simcc_x = self._normalize(simcc_x, self.num_sample, self._tau)
        simcc_y = self._normalize(simcc_y, self.num_sample, self._tau)

        if self.training:
            B, K, S, W = simcc_x.shape
            _, _, S, H = simcc_y.shape

            eps_x = torch.rand(B, K, S, W, device=feats.device)
            eps_y = torch.rand(B, K, S, H, device=feats.device)

            if self.basis_type == 'uni':
                eps_x -= 0.5
                eps_y -= 0.5

                w_x = self.linspace_x + eps_x
                w_y = self.linspace_y + eps_y

            elif self.basis_type == 'gaussian':
                eps_px = torch.exp(-eps_x**2 * 2)
                eps_py = torch.exp(-eps_y**2 * 2)

                simcc_x *= eps_px
                simcc_y *= eps_py

                simcc_x /= simcc_x.sum(dim=-1, keepdim=True)
                simcc_y /= simcc_y.sum(dim=-1, keepdim=True)

                w_x = self.linspace_x + eps_x
                w_y = self.linspace_y + eps_y

            elif self.basis_type == 'tri':
                eps_x, _ = self._uni2tri(eps_x)
                eps_y, _ = self._uni2tri(eps_y)

                w_x = self.linspace_x + eps_x
                w_y = self.linspace_y + eps_y

                simcc_x = self._retrive_p(simcc_x, w_x)
                simcc_y = self._retrive_p(simcc_y, w_y)

                simcc_x /= simcc_x.sum(dim=-1, keepdim=True)
                simcc_y /= simcc_y.sum(dim=-1, keepdim=True)

            pred_x = (simcc_x * w_x).sum(dim=-1, keepdim=True)  # B, K, S, 1
            pred_y = (simcc_y * w_y).sum(dim=-1, keepdim=True)

        else:
            pred_x = (simcc_x * self.linspace_x).sum(dim=-1, keepdim=True)
            pred_y = (simcc_y * self.linspace_y).sum(dim=-1, keepdim=True)

        pred_x /= self.linspace_x.size(2)
        pred_y /= self.linspace_y.size(2)

        if self.debias:
            C_x = simcc_x.exp().sum(dim=-1, keepdim=True)
            pred_x = C_x / (C_x - 1) * (pred_x - 1 / (2 * C_x))

            C_y = simcc_y.exp().sum(dim=-1, keepdim=True)
            pred_y = C_x / (C_y - 1) * (pred_y - 1 / (2 * C_y))

        if self.training:
            pred = torch.cat([pred_x, pred_y], dim=-1)
            return pred, output_sigma
        else:
            return torch.cat([pred_x, pred_y, output_sigma], dim=-1)

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

            _batch_coords = self.forward(_feats)
            _batch_coords[..., 2:] = _batch_coords[..., 2:].sigmoid()

            _batch_coords_flip = flip_coordinates(
                self.forward(_feats_flip),
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            _batch_coords_flip[..., 2:] = _batch_coords_flip[..., 2:].sigmoid()

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords, _ = self.forward(feats)  # (B, K, D)
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

        pred_coords, pred_sigma = self.forward(inputs)

        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = 0.
        avg_acc = 0.
        S = pred_coords.shape[2]
        for i in range(S):
            loss += self.loss_module(pred_coords[:, :, i, :], pred_sigma,
                                     keypoint_labels, keypoint_weights)

            # calculate accuracy
            _, t_avg_acc, _ = keypoint_pck_accuracy(
                pred=to_numpy(pred_coords[:, :, i, :]),
                gt=to_numpy(keypoint_labels),
                mask=to_numpy(keypoint_weights) > 0,
                thr=0.05,
                norm_factor=np.ones((pred_coords.size(0), 2),
                                    dtype=np.float32))

            avg_acc += t_avg_acc

        loss /= S
        losses.update(loss_kpt=loss)

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device) / S
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
