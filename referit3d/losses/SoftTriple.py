# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K, class_to_ignore, reduction, device):
        super(SoftTriple, self).__init__()
        # 20, 0.1, 0.2, 0.01, 64, 98, 10
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.device = device
        self.class_to_ignore = class_to_ignore
        self.reduction = reduction
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).to(self.device)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        """

        Args:
            input: torch.Size([b, dim, N_Objs])
            target: torch.Size([b, N_Objs])

        Returns:

        """
        centers = F.normalize(self.fc, p=2, dim=0).to(input.device)
        if self.cN > 2:
            input = input.permute(0, 2, 1)  # [b, N_Objs, dim]
        input = input.flatten(start_dim=0, end_dim=1)  # [b*N_Objs, dim]
        target = target.flatten()  # [b*N_Objs,]
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).to(self.device)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target, ignore_index=self.class_to_ignore,
                                       reduction=self.reduction)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
