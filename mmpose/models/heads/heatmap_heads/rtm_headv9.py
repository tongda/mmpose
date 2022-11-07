# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

# import numpy as np
import torch
# import torch.nn.functional as F
from torch import Tensor, nn

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.gilbert2d import gilbert2d
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ...utils.gau import PGAU, StarReLU
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


class SE(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        w = self.block(x).sigmoid()
        x = x * w
        return x


@MODELS.register_module()
class RTMHeadv9(BaseHead):

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        use_hilbert_flatten: bool = False,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            shift=True,
            dropout_rate=0.,
            act_fn='SiLU',
        ),
        use_decoder: bool = False,
        use_se: bool = True,
        num_enc: int = 1,
        cross_attn: bool = False,
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.align_corners = align_corners
        self.input_transform = input_transform
        self.input_index = input_index

        self.use_hilbert_flatten = use_hilbert_flatten
        self.hidden_dims = gau_cfg.hidden_dims
        self.s = gau_cfg.s
        self.shift = gau_cfg.shift
        self.dropout_rate = gau_cfg.dropout_rate
        self.drop_path = gau_cfg.drop_path
        self.num_enc = num_enc
        self.use_decoder = use_decoder
        self.cross_attn = cross_attn

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

        # cfg = dict(
        #     type='Conv2d',
        #     in_channels=in_channels,
        #     out_channels=out_channels*4,
        #     kernel_size=5,
        #     padding=2,
        #     )
        # self.final_layer = build_conv_layer(cfg)
        self.gap = nn.AdaptiveAvgPool2d(in_featuremap_size)

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        if use_hilbert_flatten:
            hilbert_mapping = []
            for x, y in gilbert2d(in_featuremap_size[1],
                                  in_featuremap_size[0]):
                hilbert_mapping.append(x * in_featuremap_size[0] + y)
            self.hilbert_mapping = hilbert_mapping

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        # self.mlp = nn.Sequential(
        #     ScaleNorm(flatten_dims),
        #     nn.Linear(flatten_dims, hidden_dims, bias=False),
        #     StarReLU(True),
        #     nn.Conv1d(in_channels, out_channels, 1, bias=False),
        #     nn.Linear(hidden_dims, hidden_dims, bias=False)
        # )
        self.channel_token_proj = PGAU(
            in_channels,
            flatten_dims,
            self.hidden_dims,
            s=flatten_dims // 2,
            dropout_rate=self.dropout_rate,
            drop_path=self.drop_path,
            shift='time' if self.shift else None,
            act_fn=self.act_fn,
            use_rel_bias=False)
        self.act = StarReLU(True)
        self.pixel_token_proj = PGAU(
            flatten_dims,
            in_channels,
            out_channels,
            s=self.s,
            dropout_rate=self.dropout_rate,
            drop_path=self.drop_path,
            shift=None,
            act_fn=self.act_fn,
            use_rel_bias=False)
        self.mlp = nn.Linear(self.hidden_dims, self.hidden_dims, bias=False)

        encoder = [
            PGAU(
                in_channels,
                self.hidden_dims,
                self.hidden_dims,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=self.drop_path,
                shift='structure' if self.shift else None,
                act_fn=self.act_fn) for _ in range(self.num_enc)
        ]
        self.encoder = nn.Sequential(*encoder)

        if use_se:
            block_x = PGAU(
                in_channels,
                self.hidden_dims,
                self.hidden_dims,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=0.,
                shift='structure' if self.shift else None,
                act_fn=self.act_fn)
            block_y = PGAU(
                in_channels,
                self.hidden_dims,
                self.hidden_dims,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=0.,
                shift='structure' if self.shift else None,
                act_fn=self.act_fn)
            self.mlp_x = SE(block_x)
            self.mlp_y = SE(block_y)
        else:
            self.mlp_x = PGAU(
                in_channels,
                self.hidden_dims,
                self.hidden_dims if self.use_decoder else W,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=self.drop_path,
                shift='structure' if self.shift else None,
                act_fn=self.act_fn)
            self.mlp_y = PGAU(
                in_channels,
                self.hidden_dims,
                self.hidden_dims if self.use_decoder else H,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=self.drop_path,
                shift='structure' if self.shift else None,
                act_fn=self.act_fn)

        if use_decoder:
            self.coord_x_token = nn.Parameter(
                torch.randn((1, W, self.hidden_dims)))
            self.coord_y_token = nn.Parameter(
                torch.randn((1, H, self.hidden_dims)))

            self.decoder_x = PGAU(
                self.out_channels,
                self.hidden_dims,
                self.hidden_dims,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=self.drop_path,
                attn_type='cross-attn' if cross_attn else 'self-attn',
                shift='structure' if self.shift else None,
                act_fn=self.act_fn)
            self.decoder_y = PGAU(
                self.out_channels,
                self.hidden_dims,
                self.hidden_dims,
                s=self.s,
                dropout_rate=self.dropout_rate,
                drop_path=self.drop_path,
                attn_type='cross-attn' if cross_attn else 'self-attn',
                shift='structure' if self.shift else None,
                act_fn=self.act_fn)

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
        feats = self.gap(feats)

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)
        if self.use_hilbert_flatten:
            feats = feats[:, :, self.hilbert_mapping]

        # B, 768, 48
        feats = self.channel_token_proj(feats)  # -> B, 768, 256
        feats = feats.permute(0, 2, 1)  # -> B, 256, 768
        feats = self.act(feats)
        feats = self.pixel_token_proj(feats)  # -> B, 256, 17
        feats = feats.permute(0, 2, 1)  # -> B, 17, 256
        feats = self.mlp(feats)

        feats = self.encoder(feats)

        pred_x = self.mlp_x(feats)
        pred_y = self.mlp_y(feats)

        if self.use_decoder:
            coord_x_token = self.coord_x_token.repeat((feats.size(0), 1, 1))
            coord_y_token = self.coord_y_token.repeat((feats.size(0), 1, 1))

            if self.cross_attn:
                pred_x = self.decoder_x((pred_x, coord_x_token, coord_x_token))
                pred_y = self.decoder_y((pred_y, coord_y_token, coord_y_token))
            else:
                pred_x = self.decoder_x(pred_x)
                pred_y = self.decoder_y(pred_y)

            pred_x = torch.bmm(pred_x, coord_x_token.permute(0, 2, 1))
            pred_y = torch.bmm(pred_y, coord_y_token.permute(0, 2, 1))

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
