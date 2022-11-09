import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle

from projects.mmdet3d_plugin.hdmaptr.utils.homography import IPM
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
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
    parser.add_argument('--samples', default=2000, help='samples to visualize')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['se_points',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range
    voxel_size = cfg.voxel_size
    xbound = [pc_range[0], pc_range[3], voxel_size[0]]
    ybound = [pc_range[1], pc_range[4], voxel_size[1]]

    ipm_func = IPM(xbound, ybound,1, N=6, C=3, z_roll_pitch=True, visual=True, extrinsic=False)

    logger.info('BEGIN vis test dataset samples gt label & pred')



    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
       
        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        lidar2global = img_metas[0]['lidar2global']
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)
        yaw_pitch_roll = lidar2global_rotation.yaw_pitch_roll
        translation = torch.tensor(lidar2global_translation).unsqueeze(0).to(
                            dtype=torch.float32)
        yaw_pitch_roll = torch.tensor(yaw_pitch_roll).unsqueeze(0).to(
                            dtype=torch.float32)
        import pdb;pdb.set_trace()
        ipm_warped_topdown = ipm_func(img.cuda(),img_metas,translation.cuda(),yaw_pitch_roll.cuda())
        ipm_warped_topdown = ipm_warped_topdown[0].permute(1,2,0).contiguous().cpu().numpy()
        ipm_warped_topdown = mmcv.imdenormalize(ipm_warped_topdown, mean, std, to_bgr=to_bgr) 

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        bev_warped_name = 'ipm_warped_topdown.jpg'
        bev_path = osp.join(sample_dir,bev_warped_name)
        mmcv.imwrite(ipm_warped_topdown, bev_path)

        img_list = [img[0,i].permute(1,2,0).contiguous().numpy() for i in range(int(img.size(1)))]
        img_list = [mmcv.imdenormalize(img, mean, std, to_bgr=to_bgr) for img in img_list]
        filename_list = img_metas[0]['filename']
        img_path_list = []
        # save cam img for sample
        for img, filename in zip(img_list, filename_list):
            filename = osp.basename(filename)
            filename_splits = filename.split('__')
            # sample_dir = filename_splits[0]
            # sample_dir = osp.join(args.show_dir, sample_dir)
            # mmcv.mkdir_or_exist(osp.abspath(sample_dir))
            img_name = filename_splits[1] + '.jpg'
            img_path = osp.join(sample_dir,img_name)
            img_path_list.append(img_path)
            mmcv.imwrite(img, img_path)

        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
