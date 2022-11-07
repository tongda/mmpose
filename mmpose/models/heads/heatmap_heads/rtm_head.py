# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer
from torch import Tensor, nn

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.gilbert2d import gilbert2d
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ...utils.rtmpose_block import SE, RTMBlock, StarReLU
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMHead(BaseHead):

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        channel_mixing: str = 'heavy',
        use_hilbert_flatten: bool = False,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            shift=True,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='StarReLU',
            use_rel_bias=False,
        ),
        use_coord_token: bool = False,
        axis_align: bool = True,
        num_self_attn: int = 1,
        use_cross_attn: bool = False,
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

        self.channel_mixing = channel_mixing
        self.use_hilbert_flatten = use_hilbert_flatten
        self.num_self_attn = num_self_attn
        self.use_coord_token = use_coord_token
        self.use_cross_attn = use_cross_attn

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

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        if channel_mixing == 'light':
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            self.final_layer = build_conv_layer(cfg)

        else:
            self.channel_token_proj = RTMBlock(
                in_channels,
                flatten_dims,
                gau_cfg.hidden_dims,
                s=flatten_dims // 2,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                shift='time' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=False)
            self.act = StarReLU(True)
            self.pixel_token_proj = RTMBlock(
                flatten_dims,
                in_channels,
                out_channels,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                shift=None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=False)
            self.mlp = nn.Linear(
                gau_cfg.hidden_dims, gau_cfg.hidden_dims, bias=False)

        if use_hilbert_flatten:
            hilbert_mapping = []
            for x, y in gilbert2d(in_featuremap_size[1],
                                  in_featuremap_size[0]):
                hilbert_mapping.append(x * in_featuremap_size[0] + y)
            self.hilbert_mapping = hilbert_mapping

        W = int(self.input_size[0] * gau_cfg.simcc_split_ratio)
        H = int(self.input_size[1] * gau_cfg.simcc_split_ratio)

        self_attn_module = [
            RTMBlock(
                in_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)
            for _ in range(self.num_self_attn)
        ]
        self.self_attn_module = nn.Sequential(*self_attn_module)

        if axis_align:
            block_x = RTMBlock(
                in_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=0.,
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)
            block_y = RTMBlock(
                in_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=0.,
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)
            self.mlp_x = SE(block_x)
            self.mlp_y = SE(block_y)
        else:
            self.mlp_x = RTMBlock(
                in_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims if self.use_coord_token else W,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)
            self.mlp_y = RTMBlock(
                in_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims if self.use_coord_token else H,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)

        if use_coord_token:
            self.coord_x_token = nn.Parameter(
                torch.randn((1, W, gau_cfg.hidden_dims)))
            self.coord_y_token = nn.Parameter(
                torch.randn((1, H, gau_cfg.hidden_dims)))

            self.decoder_x = RTMBlock(
                self.out_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                attn_type='cross-attn' if use_cross_attn else 'self-attn',
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)
            self.decoder_y = RTMBlock(
                self.out_channels,
                gau_cfg.hidden_dims,
                gau_cfg.hidden_dims,
                s=gau_cfg.s,
                dropout_rate=gau_cfg.dropout_rate,
                drop_path=gau_cfg.drop_path,
                attn_type='cross-attn' if use_cross_attn else 'self-attn',
                shift='structure' if gau_cfg.shift else None,
                act_fn=gau_cfg.act_fn,
                use_rel_bias=gau_cfg.use_rel_bias)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is multi scale feature maps and the
        output is the heatmap.
        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.
        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = self._transform_inputs(feats)

        if self.channel_mixing == 'light':
            # B, C, HW
            feats = self.final_layer(feats)  # -> B, K, hidden

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)
        if self.use_hilbert_flatten:
            feats = feats[:, :, self.hilbert_mapping]

        if self.channel_mixing == 'heavy':
            # B, C, HW
            feats = self.channel_token_proj(feats)  # -> B, C, hidden
            feats = feats.permute(0, 2, 1)  # -> B, hidden, C
            feats = self.act(feats)
            feats = self.pixel_token_proj(feats)  # -> B, hidden, K
            feats = feats.permute(0, 2, 1)  # -> B, K, hidden
            feats = self.mlp(feats)  # -> B, K, hidden

        feats = self.self_attn_module(feats)

        pred_x = self.mlp_x(feats)
        pred_y = self.mlp_y(feats)

        if self.use_coord_token:
            coord_x_token = self.coord_x_token.repeat((feats.size(0), 1, 1))
            coord_y_token = self.coord_y_token.repeat((feats.size(0), 1, 1))

            if self.use_cross_attn:
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
