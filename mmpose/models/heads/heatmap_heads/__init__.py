# Copyright (c) OpenMMLab. All rights reserved.
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .mspn_head import MSPNHead
from .rtm_head import RTMHead
from .rtm_head2 import RTMHead2
from .rtm_head3 import RTMHead3
from .simcc_head import SimCCHead
from .vipnas_head import ViPNASHead

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 'SimCCHead', 'RTMHead',
    'RTMHead2', 'RTMHead3'
]
