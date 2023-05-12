import torch
import numpy as np
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
import torch.nn as nn
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet3d.ops import bev_pool
from mmcv.runner import force_fp32, auto_fp16

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
        x = self.get_cam_feats(images)
        x = self.bev_pool(geom, x)
        # import pdb;pdb.set_trace()
        x = x.permute(0,1,3,2).contiguous()
        
        return x



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
        self.depthnet = nn.Conv2d(in_channels, int(self.D + self.C), 1)
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
    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, images, img_metas):
        x = super().forward(images, img_metas)
        x = self.downsample(x)
        return x