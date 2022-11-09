# from ..chamfer_dist import ChamferDistance
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance


def custom_polyline_score(pred_lines, gt_lines, linewidth=1., metric='chamfer'):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''
    if metric == 'iou':
        linewidth = 1.0
    positive_threshold = 1.
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = \
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                          for i in pred_lines]
    gt_lines_shapely =\
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                        for i in gt_lines]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))


    if metric=='chamfer':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric=='iou':
        iou_matrix = np.zeros((num_preds, num_gts),dtype=np.float64)
    else:
        raise NotImplementedError

    for i, pline in enumerate(gt_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                if metric=='chamfer':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    # import pdb;pdb.set_trace()
                    valid_ab = dist_mat.min(-1).mean()
                    valid_ba = dist_mat.min(-2).mean()

                    iou_matrix[pred_id, i] = -(valid_ba+valid_ab)/2
                elif metric=='iou':
                    inter = o.intersection(pline).area
                    union = o.union(pline).area
                    iou_matrix[pred_id, i] = inter / union

    return iou_matrix

if __name__ == '__main__':
    import torch

    line1 = torch.tensor([
        [1, 5], [3, 5], [5, 5]
    ])

    line0 = torch.tensor([
        [3, 6], [4, 8], [5, 6]
    ])

    line2 = torch.tensor([
        [1, 4], [3, 4], [5, 4]
    ])

    line3 = torch.tensor([
        [4, 4], [3, 3], [5, 3]
    ])

    gt = torch.stack((line2, line3), dim=0).type(torch.float32)
    pred = torch.stack((line0, line1), dim=0).type(torch.float32)

    # import ipdb; ipdb.set_trace()
    import mmcv
    # with mmcv.Timer():
    #     gt = upsampler(gt, pts=10)
    #     pred = upsampler(pred, pts=10)

    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    from descartes import PolygonPatch
    
    iou_matrix = vec_iou(pred,gt)
    print(iou_matrix)
    # import pdb;pdb.set_trace()
    score_matrix = custom_polyline_score(pred, gt, linewidth=1., metric='chamfer')
    print(score_matrix)
    fig, ax = plt.subplots()
    for i in gt:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], 'o', color='red')
        plt.plot(i[:, 0], i[:, 1], '-', color='red')

        dilated = LineString(i).buffer(1, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
        patch1 = PolygonPatch(dilated, fc='red', ec='red', alpha=0.5, zorder=-1)
        ax.add_patch(patch1)

    for i in pred:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], 'o', color='blue')
        plt.plot(i[:, 0], i[:, 1], '-', color='blue')

        dilated = LineString(i).buffer(1, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
        patch1 = PolygonPatch(dilated, fc='blue', ec='blue', alpha=0.5, zorder=-1)
        ax.add_patch(patch1)


    ax.axis('equal')


    plt.savefig('test3.png')    