# Copyright (c) OpenMMLab. All rights reserved.
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss
from .heatmap_loss import AdaptiveWingLoss
from .iou_loss import (BoundedIoULoss, CIoULoss, DIoULoss, GIoULoss, IoULoss,
                       bounded_iou_loss, iou_loss)
from .loss_wrappers import MultipleLossWrapper
from .mse_loss import (CombinedTargetIOULoss, CombinedTargetMSELoss,
                       KeypointMSELoss, KeypointOHKMMSELoss)
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'CombinedTargetMSELoss',
    'HeatmapLoss', 'AELoss', 'MultiLossFactory', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss',
    'CombinedTargetIOULoss', 'BoundedIoULoss', 'CIoULoss', 'DIoULoss',
    'GIoULoss', 'IoULoss', 'bounded_iou_loss', 'iou_loss'
]
