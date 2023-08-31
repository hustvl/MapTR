import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
import torch.nn.functional as F
from mmdet.models.losses import FocalLoss, weight_reduce_loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register_module(force=True)
class SimpleLoss_v1(nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(SimpleLoss_v1, self).__init__()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        # self.loss_fn = torch.nn.CrossEntroyLoss(reduction="none")
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        bs, pred_class_num, bev_h, bev_w = ypred.shape
        ypred = ypred.permute(0, 2, 3, 1).reshape(bs*bev_h*bev_w, pred_class_num).contiguous()
        ytgt = ytgt.view(-1)
        ytgt = F.one_hot(ytgt.long(), num_classes=pred_class_num+1).view(-1, pred_class_num+1)[:, 1:]
        fg_mask = torch.max(ytgt, dim=1).values > 0.0
        ypred = ypred[fg_mask]
        ytgt = ytgt[fg_mask]
        loss = F.binary_cross_entropy_with_logits(ypred, ytgt.float(), reduction='none',).sum() / max(1.0, fg_mask.sum())
        return loss*self.loss_weight

@LOSSES.register_module()
class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        # import ipdb;ipdb.set_trace()
        loss = self.loss_fn(ypred, ytgt)
        return loss*self.loss_weight

@LOSSES.register_module()
class MaskFocalLoss(FocalLoss):
    def __init__(self,**kwargs):
        super(MaskFocalLoss, self).__init__(**kwargs)
    
    def forward(self, 
                pred, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if not self.use_sigmoid:
            raise NotImplementedError
        
        num_classes = pred.size(1)
        loss = 0
        for index in range(num_classes):
            loss += self.loss_weight * py_sigmoid_focal_loss(
                pred[:,index],
                target[:,index],
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        # import ipdb; ipdb.set_trace()
        loss /= num_classes
        return loss