import os.path as osp
import argparse
import os
import glob
import cv2
import mmcv
CAMS = ['FRONT_LEFT','FRONT','FRONT_RIGHT',
             'BACK_LEFT','BACK','BACK_RIGHT',]
VIEWS_NAME = 'surroud_view.jpg'
GT_MAP_NAME = 'GT_fixednum_pts_MAP.png'
PRED_MAP_NAME = 'PRED_MAP_plot.png'

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('visdir', help='visualize directory')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--sample-name', default='SAMPLE_VIS.png', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    parent_dir = osp.join(args.visdir,'..')
    vis_subdir_list = []

    file_list = os.listdir(args.visdir)
    prog_bar = mmcv.ProgressBar(len(file_list))
    for file in file_list:
        file_path = osp.join(args.visdir, file) 
        if os.path.isdir(file_path):
            vis_subdir_list.append(file_path)
            sample_path = osp.join(file_path,args.sample_name)
            row_1_list = []
            for cam in CAMS[:3]:
                cam_img_name = 'CAM_'+ cam + '.jpg'
                cam_img = cv2.imread(osp.join(file_path, cam_img_name))
                # import pdb;pdb.set_trace()
                lw = 8
                tf = max(lw - 1, 1)
                w, h = cv2.getTextSize(cam, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                p1 = (0,0)
                p2 = (w,h+3)
                color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(cam_img,
                            cam, (p1[0], p1[1] + h + 2),
                            0,
                            lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                row_1_list.append(cam_img)
            row_2_list = []
            for cam in CAMS[3:]:
                cam_img_name = 'CAM_'+ cam + '.jpg'
                cam_img = cv2.imread(osp.join(file_path, cam_img_name))
                if cam == 'BACK':
                    cam_img = cv2.flip(cam_img, 1)
                # import pdb;pdb.set_trace()
                lw = 8
                tf = max(lw - 1, 1)
                w, h = cv2.getTextSize(cam, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                p1 = (0,0)
                p2 = (w,h+3)
                color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(cam_img,
                            cam, (p1[0], p1[1] + h + 2),
                            0,
                            lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                row_2_list.append(cam_img)
            row_1_img=cv2.hconcat(row_1_list)
            row_2_img=cv2.hconcat(row_2_list)
            cams_img = cv2.vconcat([row_1_img,row_2_img])


            map_img = cv2.imread(osp.join(file_path,PRED_MAP_NAME))
            gt_map_img = cv2.imread(osp.join(file_path,GT_MAP_NAME))
            map_img = cv2.copyMakeBorder(map_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
            gt_map_img = cv2.copyMakeBorder(gt_map_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)

            cams_h,cam_w,_ = cams_img.shape
            map_h,map_w,_ = map_img.shape
            resize_ratio = cams_h / map_h
            resized_w = map_w * resize_ratio
            resized_map_img = cv2.resize(map_img,(int(resized_w),int(cams_h)))
            resized_gt_map_img = cv2.resize(gt_map_img,(int(resized_w),int(cams_h)))



            sample_img = cv2.hconcat([cams_img, resized_map_img,resized_gt_map_img])
            cv2.imwrite(sample_path, sample_img)
        prog_bar.update()
    print('DONE!')


if __name__ == '__main__':
    main()

