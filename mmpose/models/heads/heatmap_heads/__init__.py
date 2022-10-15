# Copyright (c) OpenMMLab. All rights reserved.
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .mspn_head import MSPNHead
from .simcc_gap_fc_head import SimCC_GAP_FC
from .simcc_gau_head import SimCC_GAU_Head
from .simcc_head import SimCCHead
from .simcc_ipr_head import SimCC_IPR_Head
from .simcc_rle_head import SimCC_RLE_Head
from .simcc_sampling_head import SimCC_SamplingArgmax_Head
from .vipnas_head import ViPNASHead

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 'SimCCHead',
    'SimCC_RLE_Head', 'SimCC_GAP_FC', 'SimCC_IPR_Head',
    'SimCC_SamplingArgmax_Head', 'SimCC_GAU_Head'
]
