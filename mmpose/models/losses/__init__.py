# Copyright (c) OpenMMLab. All rights reserved.
from .classification_loss import (BCELoss, DistanceWeightedKLLoss,
                                  JSDiscretLoss, JSLoss, KLDiscretLoss,
                                  SimCCBCELoss, UncertainCLSLoss)
from .emd_loss import EMDLoss
from .heatmap_loss import AdaptiveWingLoss
from .iou_loss import (BoundedIoULoss, CIoULoss, DIoULoss, GIoULoss, IoULoss,
                       bounded_iou_loss, iou_loss)
from .loss_wrappers import MultipleLossWrapper
from .mse_loss import (BalancedMSELoss, CombinedTargetIOULoss,
                       CombinedTargetMSELoss, KeypointMSELoss,
                       KeypointOHKMMSELoss)
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (QFL, BoneLoss, L1Loss, MPJPELoss, MSELoss,
                              RLECLSLoss, RLELoss, SemiSupervisionLoss,
                              SmoothL1Loss, SoftWingLoss, WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'CombinedTargetMSELoss',
    'HeatmapLoss', 'AELoss', 'MultiLossFactory', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss',
    'CombinedTargetIOULoss', 'BoundedIoULoss', 'CIoULoss', 'DIoULoss',
    'GIoULoss', 'IoULoss', 'bounded_iou_loss', 'iou_loss', 'BalancedMSELoss',
    'EMDLoss', 'JSLoss', 'UncertainCLSLoss', 'RLECLSLoss',
    'DistanceWeightedKLLoss', 'QFL', 'SimCCBCELoss'
]
