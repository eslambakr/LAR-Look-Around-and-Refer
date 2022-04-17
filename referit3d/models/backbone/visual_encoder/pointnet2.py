"""
This implementation is borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from referit3d.models.backbone.visual_encoder.pointnet2_utils import PointNetSetAbstractionMsg, \
    PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True, mode="train"):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(32, [0.2], [32], in_channel, [[64, 64, 128]])
        self.sa2 = PointNetSetAbstractionMsg(16, [0.4], [32], 128, [[128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 128], True)
        self.fc1 = nn.Linear(128, num_class)

    def forward(self, xyz, mode):
        # xyz = xyz.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1).contiguous()
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm, mode)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, mode)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, mode)
        x = l3_points.view(B, 128)
        x = self.fc1(x)
        # x = F.log_softmax(x, -1)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
