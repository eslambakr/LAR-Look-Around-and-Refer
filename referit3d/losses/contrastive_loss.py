import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def contrastive_loss(self, output1, output2, target, reduction=True):
        # expand because only one anchor against N distractors
        output1 = output1.expand(output2.size(0), -1)

        # contrastive loss
        distances = (output2 - output1).pow(2).sum(1).clamp(min=1e-12).sqrt()  # squared distances
        losses = 0.5 * (target * distances +
                        (1 + -1 * target) * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if reduction else losses.sum()

    def forward(self, features, target_idx, distractors_idx, reduction=True):
        total = 0.
        features = F.normalize(features, p=2, dim=-1)

        for b_i in range(features.shape[0]):  # Loop on scenes
            total += self.contrastive_loss(features[b_i, target_idx[b_i]], features[b_i, distractors_idx[b_i].bool()],
                                           0, reduction)
        return total / b_i
