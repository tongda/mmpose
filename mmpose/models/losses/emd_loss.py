# Copyright (c) OpenMMLab. All rights reserved.
# from typing import Union

# import pykeops.torch as keops
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2`
    locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to
         the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed.
             Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, wx, wy):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function

        # both marginals are fixed with equal weights
        # mu = torch.empty(batch_size, x_points, dtype=torch.float,
        #                  requires_grad=False).fill_(1.0 / x_points).squeeze()
        # nu = torch.empty(batch_size, y_points, dtype=torch.float,
        #                  requires_grad=False).fill_(1.0 / y_points).squeeze()
        mu = wx.squeeze()
        nu = wy.squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            t1 = torch.log(mu + 1e-8)
            t2 = torch.logsumexp(self.M(C, u, v), dim=-1)
            u = self.eps * (t1 - t2) + u
            t1 = torch.log(nu + 1e-8)
            t2 = torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
            v = self.eps * (t1 - t2) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates."""
        'Mij=(−cij+ui+vj)/ϵ'
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=1):
        """Returns the matrix of |xi−yj|p."""
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin))**p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """Barycenter subroutine, used by kinetic acceleration through
        extrapolation."""
        return tau * u + (1 - tau) * u1


@MODELS.register_module()
class EMDLoss(nn.Module):

    def __init__(self, use_target_weight=False):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)

    def forward(self, preds, targets, simcc_dims, target_weight=None):
        # preds   (B, K, Wx)
        # targets (B, K)
        B, K, _ = preds.shape
        relu_preds = F.relu(preds)
        preds = relu_preds / relu_preds.sum(dim=-1, keepdims=True)

        d1 = targets.long().clamp(0, simcc_dims - 1)
        d2 = (d1 + 1).clamp(0, simcc_dims - 1)

        t1 = d2 - targets  # B, K, 1
        t2 = targets - d1  # B, K, 1

        # w = torch.cat([t1, t2], dim=2)  # B, K, 2
        w = torch.zeros_like(preds)
        # print(w.shape)
        # print(t1.shape)
        # print(d1.shape)
        for b in range(B):
            for k in range(K):
                w[b, k, d1[b, k]] = t1[b, k]
                w[b, k, d2[b, k]] = t2[b, k]

        # w[d1] = t1
        # w[d2] = t2
        # print(w)

        x = torch.arange(
            simcc_dims, dtype=torch.float,
            device=preds.device).unsqueeze(-1)  # Wx, 1
        # y = torch.arange(2).unsqueeze(-1)  # 2, 1
        # y = torch.cat([d1, d2], dim=2)  # B, K, 2
        y = torch.arange(
            simcc_dims, dtype=torch.float,
            device=preds.device).unsqueeze(-1)  # Wx, 1

        # loss = 0.
        # for b in range(preds.size(0)):
        #     for k in range(preds.size(1)):
        #         w_x = preds[b, k]  # Wx,
        #         w_y = w[b, k]  # 2,
        #         t_loss, _, _ = sinkhorn(x, y[b, k], p=1, w_x=w_x, w_y=w_y)
        #         if target_weight is not None:
        #             t_loss *= target_weight[b, k]
        #         loss += t_loss  # Wx, 1

        p = x[None, :].repeat((B * K, 1, 1))  # B*K, Wx, 1
        q = y[None, :].repeat((B * K, 1, 1))  # B*K, Wx, 1
        # q = y.reshape(-1, y.size(-1), 1)  # B*K, 2, 1
        wx = preds.reshape(-1, preds.size(-1), 1)  # B*K, Wx, 1
        wy = w.reshape(-1, w.size(-1), 1)  # B*K, 2, 1
        # print('p', p.shape, 'q', q.shape, 'wx', wx.shape,'wy',wy.shape)
        loss, _, _ = self.sinkhorn(p, q, wx, wy)

        return loss / B / K
