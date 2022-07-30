import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# from referit3d.models.backbone.visual_encoder.pointnet2.pointnet2_modules import PointnetFPModule
from referit3d.external_tools.pointnet2.pointnet2_modules import PointnetSAModuleVotes


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, num_class, input_feature_dim=0, mode="train"):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        # --------- 4 SET ABSTRACTION LAYERS ---------
        self.sa1 = PointnetSAModuleVotes(
            npoint=32,
            radius=0.2,
            nsample=32,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=16,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 256, 512, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fc1 = nn.Linear(128, num_class)

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, mode="train"):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        # print("xyz 1 ", xyz.shape)  # [?, ?, ?]
        # print("features 1 ", features.shape)  # [?, ?, ?]
        xyz, features, fps_inds = self.sa1(xyz, features)
        # print("xyz out 1 ", xyz.shape)  # [?, ?, ?]
        # print("features out 1 ", features.shape)  # [?, ?, ?]
        # print("fps_inds out 1 ", fps_inds.shape)  # [?, ?, ?]

        xyz, features, fps_inds = self.sa2(xyz, features)

        xyz, features, fps_inds = self.sa3(xyz, features)

        x = features.view(batch_size, 128)
        x = self.fc1(x)

        return x


if __name__ == '__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16, 20000, 6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
