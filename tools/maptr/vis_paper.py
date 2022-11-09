import argparse
import mmcv
import os
import shutil
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
import cv2

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]

CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]

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
    parser.add_argument('--score-thresh', default=0.6, type=float, help='samples to visualize')
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

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
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
    colors_plt = ['orange', 'b', 'g']


    logger.info('BEGIN vis test dataset samples gt label & pred')



    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue
       
        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        # import pdb;pdb.set_trace()
        if pts_filename not in CANDIDATE:
            continue

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        filename_list = img_metas[0]['filename']
        img_path_dict = {}
        # save cam img for sample
        for filepath in filename_list:
            filename = osp.basename(filepath)
            filename_splits = filename.split('__')
            # sample_dir = filename_splits[0]
            # sample_dir = osp.join(args.show_dir, sample_dir)
            # mmcv.mkdir_or_exist(osp.abspath(sample_dir))
            img_name = filename_splits[1] + '.jpg'
            img_path = osp.join(sample_dir,img_name)
            # img_path_list.append(img_path)
            shutil.copyfile(filepath,img_path)
            img_path_dict[filename_splits[1]] = img_path
         
        # surrounding view
        row_1_list = []
        for cam in CAMS[:3]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_1_list.append(cam_img)
        row_2_list = []
        for cam in CAMS[3:]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_2_list.append(cam_img)
        row_1_img=cv2.hconcat(row_1_list)
        row_2_img=cv2.hconcat(row_2_list)
        cams_img = cv2.vconcat([row_1_img,row_2_img])
        cams_img_path = osp.join(sample_dir,'surroud_view.jpg')
        cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
        
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
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')
                plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', format='png',dpi=1200)
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

                gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_pts_MAP.png')
                plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()           

            else: 
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')
        # import pdb;pdb.set_trace()
        # plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        # map_path = osp.join(sample_dir, 'GT_MAP.png')
        # plt.savefig(map_path, bbox_inches='tight', dpi=400)
        # plt.close()

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
        keep = scores_3d > args.score_thresh
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):
            # pts = pred_bbox_3d.reshape(-1,2)
            # x = np.array([pt[0] for pt in pts])
            # y = np.array([pt[1] for pt in pts])
            # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])
            
            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)
            # plt.quiver(pts_x[:-1], pts_y[:-1], pts_x[1:] - pts_x[:-1], pts_y[1:] - pts_y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[pred_label_3d])


            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)
            # import pdb;pdb.set_trace()
            # plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[pred_label_3d],facecolor='none'))
            # plt.text(pts_x[0], pts_y[0], s,  fontsize=2)


        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        

        # merge surr_view, pred, gt
        # cams_img_path
        # gt_fixedpts_map_path
        # map_path
        # fig = plt.figure(constrained_layout=False,figsize=(16, 3), facecolor='white')
        # gs = fig.add_gridspec(1, 4)
        # gs.update(wspace=0, hspace=0) 
        # axs = [
        #     fig.add_subplot(gs[0, 2]),
        #     fig.add_subplot(gs[0, 3]),
        # ]
        # fleft_ax = fig.add_subplot(gs[:, 0:2])
        # cams_img = Image.open(cams_img_path)
        # fleft_ax.imshow(cams_img)
        # fleft_ax.axis('off')
        # fleft_ax.set_title('surrounding views')
        # pred_map =  Image.open(map_path)
        # # import pdb;pdb.set_trace()
        # box0 = axs[0].get_position()
        # box0.x0 = box0.x0-0.2
        # box0.x1 = box0.x1-0.2
        # axs[0].set_position(box0)
        # axs[0].imshow(pred_map)
        # axs[0].axis('off')
        # axs[0].set_title('pred map')
        # GT_map =  Image.open(gt_fixedpts_map_path)
        # axs[1].imshow(GT_map)
        # axs[1].axis('off')
        # axs[1].set_title('GT map')
        # # gs.tight_layout(fig)
        # plt.subplots_adjust(wspace=0, hspace=0)
        # sample_vis_path = osp.join(sample_dir, 'SAMPLE_VIS.png')
        # fig.savefig(sample_vis_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=600)
        # import pdb;pdb.set_trace()
        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
