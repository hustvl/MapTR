import mmcv
import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from .tpfp_chamfer import custom_polyline_score
from shapely.geometry import LineString, Polygon


def custom_tpfp_gen(gen_lines,
             gt_lines,
             threshold=0.5,
             metric='chamfer'):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if metric == 'chamfer':
        if threshold >0:
            threshold= -threshold
    # else:
    #     raise NotImplementedError

    # import pdb;pdb.set_trace()
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]
    
    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp
    
    if num_gens == 0:
        return tp, fp
    
    gen_scores = gen_lines[:,-1] # n
    # distance matrix: n x m

    matrix = custom_polyline_score(
            gen_lines[:,:-1].reshape(num_gens,-1,2), 
            gt_lines.reshape(num_gts,-1,2),linewidth=2.,metric=metric)
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp

