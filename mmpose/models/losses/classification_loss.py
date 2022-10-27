# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.binary_cross_entropy
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target, reduction='none')
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight).mean()
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(
        self,
        use_target_weight=True,
        size_average: bool = True,
    ):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim

            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)

        if self.size_average:
            loss /= len(gt_hm)

        return loss.sum()


@MODELS.register_module()
class JSLoss(nn.Module):
    """Discrete JS Divergence loss for SimCC with Gaussian Label Smoothing.

    Modified from `the official implementation
    <https://github.com/leeyegy/SimCC>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, use_target_weight=True, beta=1.0, use_softmax=False):
        super(KLDiscretLoss, self).__init__()

        self.use_target_weight = use_target_weight
        self.beta = beta
        self.use_softmax = use_softmax

        # self.log_softmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def criterion(self, dec_outs, labels):
        scores = F.softmax(dec_outs * self.beta, dim=1)
        if self.use_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.js(scores, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): _description_
            gt_simcc (Tuple[Tensor, Tensor]): _description_
            target_weight (Tensor): _description_
        """
        output_x, output_y = pred_simcc
        target_x, target_y = gt_simcc
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight[:, idx].squeeze()
            else:
                weight = 1.

            loss += (
                self.criterion(coord_x_pred, coord_x_gt).mul(weight).sum())
            loss += (
                self.criterion(coord_y_pred, coord_y_gt).mul(weight).sum())

        return loss / num_joints


@MODELS.register_module()
class SimOTALoss(nn.Module):

    def __init__(self, use_target_weight=True, beta=1.0, use_softmax=False):
        super(SimOTALoss, self).__init__()

        self.use_target_weight = use_target_weight
        self.beta = beta
        self.use_softmax = use_softmax

        self.log_softmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.log_softmax(dec_outs * self.beta)
        if self.use_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = self.kl_loss(scores, labels)  # B, K, Wx
        return loss

    def forward(self, pred_simcc, gt_simcc, gt_coords, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): _description_
            gt_simcc (Tuple[Tensor, Tensor]): _description_
            target_weight (Tensor): _description_
        """
        output_x, output_y = pred_simcc
        target_x, target_y = gt_simcc
        coord_x, coord_y = gt_coords[:, :, 0:1], gt_coords[:, :, 1:2]
        num_joints = output_x.size(1)
        lin_x = torch.arange(
            target_x.size(-1), device=output_x.device).reshape(1,
                                                               -1)  # 1, 1, Wx
        lin_y = torch.arange(
            target_y.size(-1), device=output_y.device).reshape(1,
                                                               -1)  # 1, 1, Wy
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight[:, idx].squeeze()
            else:
                weight = 1.

            # B, 1 - 1, Wx  ->  B, Wx
            wx = 1 / (1 + torch.abs(lin_x - coord_x))
            loss_x = self.criterion(coord_x_pred, coord_x_gt)  # B, Wx
            loss_x *= weight * wx  # B, Wx

            # B, 1 - 1, Wx  ->  B, Wx
            wy = 1 / (1 + torch.abs(lin_y - coord_y))
            loss_y = self.criterion(coord_y_pred, coord_y_gt)  # B, Wx
            loss_y *= weight * wy  # B, Wx

            loss += (loss_x.sum() + loss_y.sum()) * 0.5

        return loss / num_joints


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.

    Modified from `the official implementation
    <https://github.com/leeyegy/SimCC>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, use_target_weight=True, beta=1.0, use_softmax=False):
        super(KLDiscretLoss, self).__init__()

        self.use_target_weight = use_target_weight
        self.beta = beta
        self.use_softmax = use_softmax

        self.log_softmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.log_softmax(dec_outs * self.beta)
        if self.use_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.kl_loss(scores, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): _description_
            gt_simcc (Tuple[Tensor, Tensor]): _description_
            target_weight (Tensor): _description_
        """
        output_x, output_y = pred_simcc
        target_x, target_y = gt_simcc
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight[:, idx].squeeze()
            else:
                weight = 1.

            loss += (
                self.criterion(coord_x_pred, coord_x_gt).mul(weight).sum())
            loss += (
                self.criterion(coord_y_pred, coord_y_gt).mul(weight).sum())

        return loss / num_joints
