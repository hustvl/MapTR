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

from shapely.geometry import (MultiPoint, LinearRing,
                        LineString,Polygon)


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
    parser.add_argument('checkpoint', help='checkpoint file')
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
                                'vis_convex_hull_pred')
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

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

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
    colors_plt = ['r', 'b', 'g']


    logger.info('BEGIN vis test dataset samples gt label & pred')



    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # import pdb;pdb.set_trace()
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        if args.show_cam:
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
            
            # import pdb;pdb.set_trace()
            lidar2global_translation = data['lidar2global_translation'].data[0][0]
            # visualize label on CAM
            for cam_img_path, lidar2img in zip(img_path_list, img_metas[0]['lidar2img']):
                img = Image.open(cam_img_path)
                plt.figure(figsize=(9, 16))
                fig = plt.imshow(img)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                # plt.xlim(img_metas[0]['img_shape'][0][1], 0)
                plt.ylim(img_metas[0]['img_shape'][0][0], 0)
                plt.xlim(0,img_metas[0]['img_shape'][0][1])
                # plt.ylim(0,img_metas[0]['img_shape'][0][0])
                plt.axis('off')
                for gt_bbox_3d, gt_label_3d in zip(gt_bboxes_3d[0], gt_labels_3d[0]):
                    pts = gt_bbox_3d.reshape(-1,2)
                    pts_num = pts.size(0)
                    zeros = np.zeros((pts_num, 1))
                    zeros[:] = -lidar2global_translation[-1]
                    ones = np.ones((pts_num, 1))

                    lidar_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
                    pix_coords = perspective(lidar_coords, lidar2img)
                    x = np.array([pts[0] for pts in pix_coords])
                    y = np.array([pts[1] for pts in pix_coords])

                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy',
                            angles='xy', scale=1, color=colors_plt[gt_label_3d])
                plt.savefig(cam_img_path, bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()
            # import pdb;pdb.set_trace()


        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        # import pdb;pdb.set_trace()
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
                gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    # import pdb;pdb.set_trace() 
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
            elif vis_format == 'polyline_pts':
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                # import pdb;pdb.set_trace()
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
            else: 
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'GT_MAP.jpg')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()

        # import pdb;pdb.set_trace()
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d, boxes_3d,labels_3d, pts_3d):
            # pts = pred_bbox_3d.reshape(-1,2)
            # x = np.array([pt[0] for pt in pts])
            # y = np.array([pt[1] for pt in pts])
            # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])
            
            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            # plt.scatter(pts_x, pts_y, s=1,color=colors_plt[pred_label_3d])
            plt.quiver(pts_x[:-1], pts_y[:-1], pts_x[1:] - pts_x[:-1], pts_y[1:] - pts_y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])
            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)
            # import pdb;pdb.set_trace()
            plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[pred_label_3d],facecolor='none'))
            plt.text(xy[0], xy[1], s,  fontsize=2)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_MAP.jpg')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()
        # import pdb;pdb.set_trace()

        # vis convex hull pred
        # visualize pred
        # import pdb;pdb.set_trace()
        import pdb;pdb.set_trace()
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        # result_dic = result[0]['pts_bbox']
        # boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        # scores_3d = result_dic['scores_3d']
        # labels_3d = result_dic['labels_3d']
        # pts_3d = result_dic['pts_3d']
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d, boxes_3d,labels_3d, pts_3d):
            # pts = pred_bbox_3d.reshape(-1,2)
            # x = np.array([pt[0] for pt in pts])
            # y = np.array([pt[1] for pt in pts])
            # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])
            
            pred_pts_3d = pred_pts_3d.numpy()
            # import pdb;pdb.set_trace()
            pred_geom = MultiPoint(pred_pts_3d)
            pred_convex_hull = pred_geom.convex_hull
            if pred_convex_hull.geom_type == 'LineString':
                pred_pts = np.array(list(pred_convex_hull.coords))
            elif pred_convex_hull.geom_type == 'Polygon':
                pred_pts = np.array(list(pred_convex_hull.exterior.coords))


            pts_x = pred_pts[:,0]
            pts_y = pred_pts[:,1]
            # plt.scatter(pts_x, pts_y, s=1,color=colors_plt[pred_label_3d])
            plt.quiver(pts_x[:-1], pts_y[:-1], pts_x[1:] - pts_x[:-1], pts_y[1:] - pts_y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])
            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)
            # import pdb;pdb.set_trace()
            plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[pred_label_3d],facecolor='none'))
            plt.text(xy[0], xy[1], s,  fontsize=2)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_ConvexHull_MAP.jpg')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()
        # import pdb;pdb.set_trace()



        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
