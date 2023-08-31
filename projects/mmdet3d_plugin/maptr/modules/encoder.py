import torch
import numpy as np
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
import torch.nn as nn
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet3d.ops import bev_pool
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmcv.runner import force_fp32, auto_fp16
from torch.cuda.amp.autocast_mode import autocast
from mmcv.cnn import build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F

from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor(
        [int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BaseTransform(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
    ):
        super(BaseTransform, self).__init__()
        self.in_channels = in_channels
        self.feat_down_sample = feat_down_sample
        # self.image_size = image_size
        # self.feature_size = feature_size
        self.xbound = [pc_range[0],pc_range[3], voxel_size[0]]
        self.ybound = [pc_range[1],pc_range[4], voxel_size[1]]
        self.zbound = [pc_range[2],pc_range[5], voxel_size[2]]
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = None
        self.D = int((dbound[1] - dbound[0]) / dbound[2])
        # self.frustum = self.create_frustum()
        # self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self,fH,fW,img_metas):
        # iH, iW = self.image_size
        # fH, fW = self.feature_size
        iH = img_metas[0]['img_shape'][0][0]
        iW = img_metas[0]['img_shape'][0][1]
        assert iH // self.feat_down_sample == fH
        # import pdb;pdb.set_trace()
        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        # return nn.Parameter(frustum, requires_grad=False)
        return frustum
    @force_fp32()
    def get_geometry_v1(
        self,
        fH,
        fW,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        img_metas,
        **kwargs,
    ):
        B, N, _ = trans.shape
        device = trans.device
        if self.frustum == None:
            self.frustum = self.create_frustum(fH,fW,img_metas)
            self.frustum = self.frustum.to(device)
            # self.D = self.frustum.shape[0]
        
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        # ego_to_lidar
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    @force_fp32()
    def get_geometry(
        self,
        fH,
        fW,
        lidar2img,
        img_metas,
    ):
        B, N, _, _ = lidar2img.shape
        device = lidar2img.device
        # import pdb;pdb.set_trace()
        if self.frustum == None:
            self.frustum = self.create_frustum(fH,fW,img_metas)
            self.frustum = self.frustum.to(device)
            # self.D = self.frustum.shape[0]
        
        points = self.frustum.view(1,1,self.D, fH, fW, 3) \
                 .repeat(B,N,1,1,1,1)
        lidar2img = lidar2img.view(B,N,1,1,1,4,4)
        # img2lidar = torch.inverse(lidar2img)
        points = torch.cat(
            (points, torch.ones_like(points[..., :1])), -1)
        points = torch.linalg.solve(lidar2img.to(torch.float32), 
                                    points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        # points = torch.matmul(img2lidar.to(torch.float32),
        #                       points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        # import pdb;pdb.set_trace()
        eps = 1e-5
        points = points[..., 0:3] / torch.maximum(
            points[..., 3:4], torch.ones_like(points[..., 3:4]) * eps)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran, bda):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        images,
        img_metas
    ):
        B, N, C, fH, fW = images.shape
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            camera2ego.append(img_meta['camera2ego'])
            camera_intrinsics.append(img_meta['camera_intrinsics'])
            img_aug_matrix.append(img_meta['img_aug_matrix'])
            lidar2ego.append(img_meta['lidar2ego'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = images.new_tensor(lidar2img)  # (B, N, 4, 4)
        camera2ego = np.asarray(camera2ego)
        camera2ego = images.new_tensor(camera2ego)  # (B, N, 4, 4)
        camera_intrinsics = np.asarray(camera_intrinsics)
        camera_intrinsics = images.new_tensor(camera_intrinsics)  # (B, N, 4, 4)
        img_aug_matrix = np.asarray(img_aug_matrix)
        img_aug_matrix = images.new_tensor(img_aug_matrix)  # (B, N, 4, 4)
        lidar2ego = np.asarray(lidar2ego)
        lidar2ego = images.new_tensor(lidar2ego)  # (B, N, 4, 4)

        # import pdb;pdb.set_trace()
        # lidar2cam = torch.linalg.solve(camera2ego, lidar2ego.view(B,1,4,4).repeat(1,N,1,1))
        # lidar2oriimg = torch.matmul(camera_intrinsics,lidar2cam)
        # mylidar2img = torch.matmul(img_aug_matrix,lidar2oriimg)



        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]

        # tmpgeom = self.get_geometry(
        #     fH,
        #     fW,
        #     mylidar2img,
        #     img_metas,
        # )

        geom = self.get_geometry_v1(
            fH,
            fW,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
            img_metas
        )
        mlp_input = self.get_mlp_input(camera2ego, camera_intrinsics, post_rots, post_trans)
        x, depth = self.get_cam_feats(images, mlp_input)
        x = self.bev_pool(geom, x)
        # import pdb;pdb.set_trace()
        x = x.permute(0,1,3,2).contiguous()
        
        return x, depth


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BaseTransformV2(BaseModule):
    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
        sid=False,
    ):
        super(BaseTransformV2, self).__init__()
        self.in_channels = in_channels
        self.feat_down_sample = feat_down_sample
        # self.image_size = image_size
        # self.feature_size = feature_size
        
        xbound = [pc_range[0],pc_range[3], voxel_size[0]]
        ybound = [pc_range[1],pc_range[4], voxel_size[1]]
        zbound = [pc_range[2],pc_range[5], voxel_size[2]]
        grid_config = [xbound, ybound, zbound]
        self.create_grid_infos(*grid_config)
        self.dbound = dbound
        self.sid = sid
        self.frustum = self.create_frustum(dbound,
                                    input_size, feat_down_sample)
        self.C = out_channels
        self.D = round((dbound[1] - dbound[0]) / dbound[2])
        self.fp16_enabled = False
        
    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    # @force_fp32()
    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        return torch.stack((x, y, d), -1)

    def get_lidar_coor(self, 
                       fH, 
                       fW, 
                       rots,
                       trans,
                       intrins,
                       post_rots, 
                       post_trans,
                       lidar2ego_rots,
                       lidar2ego_trans,
                       img_metas):
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        # ego_to_lidar
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
        return points

    @force_fp32()
    def get_geometry_v1(
        self,
        fH,
        fW,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        img_metas,
        **kwargs,
    ):
        B, N, _ = trans.shape
        device = trans.device
        # if self.frustum == None:
        #     self.frustum = self.create_frustum(fH,fW,img_metas)
        #     self.frustum = self.frustum.to(device)
        #     # self.D = self.frustum.shape[0]
        
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(device)- post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        # ego_to_lidar
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    @force_fp32()
    def get_geometry(
        self,
        fH,
        fW,
        lidar2img,
        img_metas,
    ):
        B, N, _, _ = lidar2img.shape
        device = lidar2img.device
        if self.frustum == None:
            self.frustum = self.create_frustum(fH,fW,img_metas)
            self.frustum = self.frustum.to(device)
            # self.D = self.frustum.shape[0]
        
        points = self.frustum.view(1,1,self.D, fH, fW, 3) \
                 .repeat(B,N,1,1,1,1)
        lidar2img = lidar2img.view(B,N,1,1,1,4,4)
        # img2lidar = torch.inverse(lidar2img)
        points = torch.cat(
            (points, torch.ones_like(points[..., :1])), -1)
        points = torch.linalg.solve(lidar2img.to(torch.float32), 
                                    points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        # points = torch.matmul(img2lidar.to(torch.float32),
        #                       points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        eps = 1e-5
        points = points[..., 0:3] / torch.maximum(
            points[..., 3:4], torch.ones_like(points[..., 3:4]) * eps)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran, bda):
        raise NotImplementedError


    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()


    @force_fp32()
    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        # if self.collapse_z:
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat
    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)
        
        return final


    @force_fp32()
    def forward(
        self,
        images,
        img_metas
    ):
        B, N, C, fH, fW = images.shape
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            camera2ego.append(img_meta['camera2ego'])
            camera_intrinsics.append(img_meta['camera_intrinsics'])
            img_aug_matrix.append(img_meta['img_aug_matrix'])
            lidar2ego.append(img_meta['lidar2ego'])

        lidar2img = np.asarray(lidar2img)
        lidar2img = images.new_tensor(lidar2img)  # (B, N, 4, 4)
        camera2ego = np.asarray(camera2ego)
        camera2ego = images.new_tensor(camera2ego)  # (B, N, 4, 4)
        camera_intrinsics = np.asarray(camera_intrinsics)
        camera_intrinsics = images.new_tensor(camera_intrinsics)  # (B, N, 4, 4)
        img_aug_matrix = np.asarray(img_aug_matrix)
        img_aug_matrix = images.new_tensor(img_aug_matrix)  # (B, N, 4, 4)
        lidar2ego = np.asarray(lidar2ego)
        lidar2ego = images.new_tensor(lidar2ego)  # (B, N, 4, 4)

        # lidar2cam = torch.linalg.solve(camera2ego, lidar2ego.view(B,1,4,4).repeat(1,N,1,1))
        # lidar2oriimg = torch.matmul(camera_intrinsics,lidar2cam)
        # mylidar2img = torch.matmul(img_aug_matrix,lidar2oriimg)

        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]

        sensor_config = [fH, fW, rots, trans, intrins, post_rots, post_trans, lidar2ego_rots, lidar2ego_trans, img_metas]

        # coor = self.get_lidar_coor(*sensor_config)
        # # tmpgeom = self.get_geometry(
        # #     fH,
        # #     fW,
        # #     mylidar2img,
        # #     img_metas,
        # # )

        coor = self.get_geometry_v1(
            fH,
            fW,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
            img_metas
        )
        mlp_input = self.get_mlp_input(camera2ego, camera_intrinsics, post_rots, post_trans)
        tran_feat, depth = self.get_cam_feats(images, mlp_input)

        bev_feat = self.voxel_pooling_v2(
            coor, depth,
            tran_feat)
        # x = self.bev_pool(geom, x)
        # import ipdb;ipdb.set_trace()
        # bev_feat = bev_feat.permute(0,1,3,2).contiguous()
        
        return bev_feat, depth



class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 aspp_mid_channels=-1,
                 only_depth=False):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.only_depth = only_depth or context_channels == 0
        if not self.only_depth:
            self.context_conv = nn.Conv2d(
                mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.context_mlp = Mlp(22, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        if not self.only_depth:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        if not self.only_depth:
            return torch.cat([depth, context], dim=1)
        else:
            return depth



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderDepth(BEVFormerEncoder):

    def __init__(self, *args, in_channels=256, out_channels=256, feat_down_sample=32, loss_depth_weight = 3.0,
                 depthnet_cfg=dict(),grid_config=None,**kwargs):

        super(BEVFormerEncoderDepth, self).__init__(*args, **kwargs)
        
        self.fp16_enabled = False

        self.loss_depth_weight = loss_depth_weight
        self.feat_down_sample = feat_down_sample
        self.grid_config = grid_config
        self.D = int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2])
        self.depth_net = DepthNet(in_channels, in_channels,
                                  0, self.D, **depthnet_cfg)


    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                mlvl_feats=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        
        bev_embed = super().forward(            
            bev_query,
            key,
            value,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs)
        # import ipdb; ipdb.set_trace()
        images = mlvl_feats[0]
        img_metas = kwargs['img_metas']
        B, N, C, fH, fW = images.shape
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            camera2ego.append(img_meta['camera2ego'])
            camera_intrinsics.append(img_meta['camera_intrinsics'])
            img_aug_matrix.append(img_meta['img_aug_matrix'])
            lidar2ego.append(img_meta['lidar2ego'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = images.new_tensor(lidar2img)  # (B, N, 4, 4)
        camera2ego = np.asarray(camera2ego)
        camera2ego = images.new_tensor(camera2ego)  # (B, N, 4, 4)
        camera_intrinsics = np.asarray(camera_intrinsics)
        camera_intrinsics = images.new_tensor(camera_intrinsics)  # (B, N, 4, 4)
        img_aug_matrix = np.asarray(img_aug_matrix)
        img_aug_matrix = images.new_tensor(img_aug_matrix)  # (B, N, 4, 4)
        lidar2ego = np.asarray(lidar2ego)
        lidar2ego = images.new_tensor(lidar2ego)  # (B, N, 4, 4)

        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]

        mlp_input = self.get_mlp_input(camera2ego, camera_intrinsics, post_rots, post_trans)
        depth = self.get_cam_feats(images, mlp_input)
        ret_dict = dict(
            bev=bev_embed['bev'],
            depth=depth,
        )
        # import ipdb; ipdb.set_trace()
        return ret_dict

    @force_fp32()
    def get_cam_feats(self, x, mlp_input):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depth_net(x, mlp_input)
        depth = x[:, : self.D].softmax(dim=1)
        depth = depth.view(B, N, self.D, fH, fW)
        return depth
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   self.feat_down_sample, W // self.feat_down_sample,
                                   self.feat_down_sample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous() 
        gt_depths = gt_depths.view(-1, self.feat_down_sample * self.feat_down_sample)
        # 把gt_depth做feat_down_sample倍数的采样
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        # 因为深度很稀疏，大部分的点都是0，所以把0变成10000，下一步取-1维度上的最小就是深度的值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   W // self.feat_down_sample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] - 
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        # import pdb;pdb.set_trace()
        if depth_preds is None:
            return 0
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)
        # fg_mask = torch.max(depth_labels, dim=1).values > 0.0 # 只计算有深度的前景的深度loss
        # import pdb;pdb.set_trace()
        fg_mask = depth_labels > 0.0 # 只计算有深度的前景的深度loss
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        # if depth_loss <= 0.:
        #     import pdb;pdb.set_trace()
        return self.loss_depth_weight * depth_loss

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        B, N, _, _ = sensor2ego.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
        downsample=1,
        loss_depth_weight = 3.0,
        depthnet_cfg=dict(),
        grid_config=None,
    ):
        super(LSSTransform, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
        )
        # import pdb;pdb.set_trace()
        self.loss_depth_weight = loss_depth_weight
        self.grid_config = grid_config
        self.depth_net = DepthNet(in_channels, in_channels,
                                  self.C, self.D, **depthnet_cfg)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, mlp_input):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depth_net(x, mlp_input)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, fH, fW)
        return x, depth

    def forward(self, images, img_metas):
        x, depth = super().forward(images, img_metas)
        x = self.downsample(x)
        ret_dict = dict(
            bev=x,
            depth=depth,
        )
        return ret_dict

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   self.feat_down_sample, W // self.feat_down_sample,
                                   self.feat_down_sample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous() 
        gt_depths = gt_depths.view(-1, self.feat_down_sample * self.feat_down_sample)
        # 把gt_depth做feat_down_sample倍数的采样
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        # 因为深度很稀疏，大部分的点都是0，所以把0变成10000，下一步取-1维度上的最小就是深度的值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   W // self.feat_down_sample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] - 
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        # import pdb;pdb.set_trace()
        if depth_preds is None:
            return 0
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)
        # fg_mask = torch.max(depth_labels, dim=1).values > 0.0 # 只计算有深度的前景的深度loss
        # import pdb;pdb.set_trace()
        fg_mask = depth_labels > 0.0 # 只计算有深度的前景的深度loss
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        # if depth_loss <= 0.:
        #     import pdb;pdb.set_trace()
        return self.loss_depth_weight * depth_loss

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        B, N, _, _ = sensor2ego.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LSSTransformV2(BaseTransformV2):
    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
        downsample=1,
        loss_depth_weight = 3.0,
        depthnet_cfg=dict(),
        grid_config = None,
        sid=False,
    ):
        super(LSSTransformV2, self).__init__(
            input_size=input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
            sid=sid,
        )
        self.loss_depth_weight = loss_depth_weight
        self.grid_config = grid_config
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.C, self.D, **depthnet_cfg)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, mlp_input):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depth_net(x, mlp_input)
        depth = x[:, : self.D].softmax(dim=1)
        tran_feat = x[:, self.D : (self.D + self.C)]

        tran_feat = tran_feat.view(B, N, self.C, fH, fW)
        # x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, fH, fW)
        return tran_feat, depth

    def forward(self, images, img_metas):
        x, depth = super().forward(images, img_metas)
        x = self.downsample(x)
        ret_dict = dict(
            bev=x,
            depth=depth,
        )
        return ret_dict

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   self.feat_down_sample, W // self.feat_down_sample,
                                   self.feat_down_sample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous() 
        gt_depths = gt_depths.view(-1, self.feat_down_sample * self.feat_down_sample)
        # 把gt_depth做feat_down_sample倍数的采样
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        # 因为深度很稀疏，大部分的点都是0，所以把0变成10000，下一步取-1维度上的最小就是深度的值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   W // self.feat_down_sample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        # import pdb;pdb.set_trace()
        if depth_preds is None:
            return 0
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)
        # fg_mask = torch.max(depth_labels, dim=1).values > 0.0 # 只计算有深度的前景的深度loss
        # import pdb;pdb.set_trace()
        fg_mask = depth_labels > 0.0 # 只计算有深度的前景的深度loss
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        # if depth_loss <= 0.:
        #     import pdb;pdb.set_trace()
        return self.loss_depth_weight * depth_loss
        

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        B, N, _, _ = sensor2ego.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
