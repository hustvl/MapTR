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
from mmdet3d.datasets import build_dataset
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
from shapely.geometry import LineString
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from descartes import PolygonPatch
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
    dataset.is_vis_on_test = True #TODO, this is a hack
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
    # import pdb;pdb.set_trace()
    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'g']


    logger.info('BEGIN vis test dataset samples gt label')
    logger.info(f'vis gt label format: {args.gt_format}')
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        
        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]


        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]

        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))


        for vis_format in args.gt_format:
            if vis_format == 'se_pts':
                gt_line_points = gt_bboxes_3d[0].start_end_points
                for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                    pts = gt_bbox_3d.reshape(-1,2).numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
            elif vis_format == 'bbox':
                gt_lines_bbox = gt_bboxes_3d[0].bbox
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                    gt_bbox_3d = gt_bbox_3d.numpy()
                    xy = (gt_bbox_3d[0],gt_bbox_3d[1])
                    width = gt_bbox_3d[2] - gt_bbox_3d[0]
                    height = gt_bbox_3d[3] - gt_bbox_3d[1]
                    # import pdb;pdb.set_trace()
                    plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[gt_label_3d],facecolor='none'))
                    # plt.Rectangle(xy, width, height,color=colors_plt[gt_label_3d])
                # continue
            elif vis_format == 'fixed_num_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    # import pdb;pdb.set_trace() 
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')
                plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', dpi=400)
                plt.close()  
            elif vis_format == 'fixed_num_pts_torch':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                # gt_bboxes_3d[0].fixed_num=20 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points_torch
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    # import pdb;pdb.set_trace() 
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                map_path = osp.join(sample_dir, 'GT_fixednum_pts_torch_MAP.jpg')
                plt.savefig(map_path, bbox_inches='tight', dpi=400)
                plt.close()  
            elif vis_format == 'polyline_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                # import pdb;pdb.set_trace()
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_MAP.png')
                plt.savefig(gt_polyline_map_path, bbox_inches='tight', dpi=400)
                plt.close()
            elif vis_format == 'polyline_pts_quiver':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                # import pdb;pdb.set_trace()
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d],alpha=0.8,zorder=-1)

                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_quiver_MAP.png')
                plt.savefig(gt_polyline_map_path, bbox_inches='tight', dpi=400)
                plt.close()      
            elif vis_format == 'polyline_dilated_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    dilated_pts = LineString(pts).buffer(1, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                    patch1 = PolygonPatch(dilated_pts, fc=colors_plt[gt_label_3d], ec=colors_plt[gt_label_3d], alpha=0.8,zorder=-1)
                    plt.gca().add_patch(patch1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                map_path = osp.join(sample_dir, 'GT_polyline_dilated_MAP.jpg')
                plt.savefig(map_path, bbox_inches='tight', dpi=400)
                plt.close() 
            elif vis_format == 'shift_fixed_num_pts':
                # plt.close()
                gt_bboxes_3d[0].fixed_num=20 #TODO, this is a hack
                shift_gt_lines_fixed_num_pts = gt_bboxes_3d[0].shift_fixed_num_sampled_points
                # import pdb;pdb.set_trace() 
                for instance_i, (shift_gt_bbox_3d, gt_label_3d) in enumerate(zip(shift_gt_lines_fixed_num_pts, gt_labels_3d[0])):
                    # import pdb;pdb.set_trace() 
                    for shift_i, gt_bbox_3d in enumerate(shift_gt_bbox_3d):
                        plt.figure(figsize=(2, 4))
                        plt.xlim(pc_range[0], pc_range[3])
                        plt.ylim(pc_range[1], pc_range[4])
                        plt.axis('off')

                        pts = gt_bbox_3d.numpy()
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                        plt.scatter(x[0], y[0], s=5,color='yellow')

                        map_path = osp.join(sample_dir, 'GT_MAP'+'_'+str(instance_i)+'_'+str(shift_i)+'.jpg')
                        plt.savefig(map_path, bbox_inches='tight', dpi=400)
                        plt.close()
                
            elif vis_format == 'shift_fixed_num_pts_v1':
                # plt.close()
                gt_bboxes_3d[0].fixed_num=20 #TODO, this is a hack
                shift_gt_lines_fixed_num_pts = gt_bboxes_3d[0].shift_fixed_num_sampled_points_v1
                # import pdb;pdb.set_trace() 
                for instance_i, (shift_gt_bbox_3d, gt_label_3d) in enumerate(zip(shift_gt_lines_fixed_num_pts, gt_labels_3d[0])):
                    # import pdb;pdb.set_trace() 
                    for shift_i, gt_bbox_3d in enumerate(shift_gt_bbox_3d):
                        plt.figure(figsize=(2, 4))
                        plt.xlim(pc_range[0], pc_range[3])
                        plt.ylim(pc_range[1], pc_range[4])
                        plt.axis('off')

                        pts = gt_bbox_3d.numpy()
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                        plt.scatter(x[0], y[0], s=5,color='yellow')

                        map_path = osp.join(sample_dir, 'GT_MAP'+'_'+str(instance_i)+'_'+str(shift_i)+'.jpg')
                        plt.savefig(map_path, bbox_inches='tight', dpi=400)
                        plt.close()
            elif vis_format == 'shift_fixed_num_pts_v2':
                # plt.close()
                gt_bboxes_3d[0].fixed_num=20 #TODO, this is a hack
                shift_gt_lines_fixed_num_pts = gt_bboxes_3d[0].shift_fixed_num_sampled_points_v2
                # import pdb;pdb.set_trace() 
                for instance_i, (shift_gt_bbox_3d, gt_label_3d) in enumerate(zip(shift_gt_lines_fixed_num_pts, gt_labels_3d[0])):
                    # import pdb;pdb.set_trace() 
                    for shift_i, gt_bbox_3d in enumerate(shift_gt_bbox_3d):
                        plt.figure(figsize=(2, 4))
                        plt.xlim(pc_range[0], pc_range[3])
                        plt.ylim(pc_range[1], pc_range[4])
                        plt.axis('off')

                        pts = gt_bbox_3d.numpy()
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                        plt.scatter(x[0], y[0], s=5,color='yellow')

                        map_path = osp.join(sample_dir, 'GT_MAP'+'_'+str(instance_i)+'_'+str(shift_i)+'.jpg')
                        plt.savefig(map_path, bbox_inches='tight', dpi=400)
                        plt.close()
            elif vis_format == 'shift_fixed_num_pts_v3':
                # plt.close()
                gt_bboxes_3d[0].fixed_num=20 #TODO, this is a hack
                shift_gt_lines_fixed_num_pts = gt_bboxes_3d[0].shift_fixed_num_sampled_points_v3
                # import pdb;pdb.set_trace() 
                for instance_i, (shift_gt_bbox_3d, gt_label_3d) in enumerate(zip(shift_gt_lines_fixed_num_pts, gt_labels_3d[0])):
                    # import pdb;pdb.set_trace() 
                    for shift_i, gt_bbox_3d in enumerate(shift_gt_bbox_3d):
                        plt.figure(figsize=(2, 4))
                        plt.xlim(pc_range[0], pc_range[3])
                        plt.ylim(pc_range[1], pc_range[4])
                        plt.axis('off')

                        pts = gt_bbox_3d.numpy()
                        x = np.array([pt[0] for pt in pts])
                        y = np.array([pt[1] for pt in pts])
                        plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                        plt.scatter(x[0], y[0], s=5,color='yellow')

                        map_path = osp.join(sample_dir, 'GT_MAP'+'_'+str(instance_i)+'_'+str(shift_i)+'.jpg')
                        plt.savefig(map_path, bbox_inches='tight', dpi=400)
                        plt.close()
            else: 
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')

        prog_bar.update()
    logger.info('\n DONE vis train dataset samples gt label')
if __name__ == '__main__':
    main()
