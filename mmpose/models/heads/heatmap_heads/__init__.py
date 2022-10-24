# Copyright (c) OpenMMLab. All rights reserved.
from .cpm_head import CPMHead
from .gau_head import GAU_Head
from .heatmap_head import HeatmapHead
from .kcm_head import KCMHead
from .kpt_coord_head import KptCoordHead
from .mspn_head import MSPNHead
from .selfmatch_head import SelfMatchHead
from .simcc_gap_fc_head import SimCC_GAP_FC
from .simcc_gau_head import SimCC_GAU_Head
from .simcc_head import SimCCHead
from .simcc_ipr_head import SimCC_IPR_Head
from .simcc_proposal_head import SimCC_Proposal_Head
from .simcc_rle_head import SimCC_RLE_Head
from .simcc_sampling_head import SimCC_SamplingArgmax_Head
from .simkcm_head import SimKCMHead
from .simkcm_sigma_head import SimKCM_Sigma_Head
from .simtoken_head import SimTokenHead
from .vipnas_head import ViPNASHead

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 'SimCCHead',
    'SimCC_RLE_Head', 'SimCC_GAP_FC', 'SimCC_IPR_Head',
    'SimCC_SamplingArgmax_Head', 'SimCC_GAU_Head', 'GAU_Head',
    'SimCC_Proposal_Head', 'SelfMatchHead', 'KptCoordHead', 'KCMHead',
    'SimKCMHead', 'SimKCM_Sigma_Head', 'SimTokenHead'
]
