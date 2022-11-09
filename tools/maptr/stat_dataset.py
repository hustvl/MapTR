# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
import numpy as np
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.utils import collect_env, get_root_logger

import sys
import os
from os import path as osp
sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
#from tools.misc.fuse_conv_bn import fuse_module

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    logger = get_root_logger()
    cfg.data.train.map_classes=['divider', 'ped_crossing','boundary'] #TODO, this is a hack
    # cfg.num_map_classes=3 #TODO, this is a hack

    dataset = custom_build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build train data set')


    logger.info('BEGIN stat train dataset samples gt label')

    prog_bar = mmcv.ProgressBar(len(dataset))

    polyline_pts = []
    for i, data in enumerate(data_loader):
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]


        gt_lines_instance = gt_bboxes_3d[0].instance_list
        # import pdb;pdb.set_trace()
        for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
            pts = np.array(list(gt_line_instance.coords))
            polyline_pts.append(pts.shape[0])

        prog_bar.update()
    polyline_pts = np.array(polyline_pts)
    print("*-"*10 + "max polyline pts num" + "*-"*10)
    logger.info(f'Num: {polyline_pts.max()}')
    print("*-"*10 + "max polyline pts num" + "*-"*10)
    logger.info('\n DONE stat train dataset samples gt box')
if __name__ == '__main__':
    main()
