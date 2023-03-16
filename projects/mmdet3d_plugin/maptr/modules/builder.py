import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg
FUSERS = Registry("fusers")
def build_fuser(cfg):
    return FUSERS.build(cfg)