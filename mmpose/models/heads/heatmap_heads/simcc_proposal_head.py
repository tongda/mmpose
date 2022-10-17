# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import build_conv_layer
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ...utils.gau import GAUplus
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class SimCC_Proposal_Head(BaseHead):

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

        self.mlp_head_x = GAUplus(self.out_channels, flatten_dims, W)
        self.mlp_head_y = GAUplus(self.out_channels, flatten_dims, H)

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

        proposal = output_coord[:, :, :2].detach().clip(0, 1)

        simcc_x = self.mlp_head_x(x, proposal[:, :, 0:1])
        simcc_y = self.mlp_head_y(x, proposal[:, :, 1:2])

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
