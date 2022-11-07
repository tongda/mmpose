# Copyright (c) OpenMMLab. All rights reserved.
from .cpm_head import CPMHead
from .gau_head import GAU_Head
from .heatmap_head import HeatmapHead
from .kcm_head import KCMHead
from .kpt_coord_head import KptCoordHead
from .mspn_head import MSPNHead
from .rtm_head import RTMHead
from .rtm_headv2 import RTMHeadv2
from .rtm_headv3 import RTMHeadv3
from .rtm_headv4 import RTMHeadv4
from .rtm_headv5 import RTMHeadv5
from .rtm_headv6 import RTMHeadv6
from .rtm_headv7 import RTMHeadv7
from .rtm_headv8 import RTMHeadv8
from .rtm_headv9 import RTMHeadv9
from .selfmatch_head import SelfMatchHead
from .simcc_gap_fc_head import SimCC_GAP_FC
from .simcc_gau_head import SimCC_GAU_Head
from .simcc_head import SimCCHead
from .simcc_ipr_head import SimCC_IPR_Head
from .simcc_proposal_head import SimCC_Proposal_Head
from .simcc_rle_head import SimCC_RLE_Head
from .simcc_sampling_head import SimCC_SamplingArgmax_Head
from .simkcm_head import SimKCMHead
from .simkcm_sigma_head import Sigma_Head
from .simota_head import SimOTAHead
from .simtoken_head import SimTokenHead
from .vipnas_head import ViPNASHead

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 'SimCCHead',
    'SimCC_RLE_Head', 'SimCC_GAP_FC', 'SimCC_IPR_Head',
    'SimCC_SamplingArgmax_Head', 'SimCC_GAU_Head', 'GAU_Head',
    'SimCC_Proposal_Head', 'SelfMatchHead', 'KptCoordHead', 'KCMHead',
    'SimKCMHead', 'Sigma_Head', 'SimTokenHead', 'SimOTAHead', 'RTMHead',
    'RTMHeadv2', 'RTMHeadv3', 'RTMHeadv4', 'RTMHeadv5', 'RTMHeadv6',
    'RTMHeadv7', 'RTMHeadv8', 'RTMHeadv9'
]
