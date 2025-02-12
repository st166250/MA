import math
from argparse import ArgumentParser
import sys
import os
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler


#sys.path.append("/home/students/studhoene1/imagequality/")
#script_dir = os.path.dirname(os.path.abspath(__file__))
#shared_dir = os.path.join(script_dir, '..',)
#sys.path.append(os.path.abspath(shared_dir))
from backbone.resnet import resnet50 ,resnet18


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


""
class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


""
class SimCLR(nn.Module):
    def __init__(
        self,
        arch: str = "resnet50",
        hidden_mlp: int = 2048, #512 for resnet18
        feat_dim: int = 128,
        first_conv: bool = True,
        maxpool1: bool = True,
        **kwargs
    ):

        super().__init__()

        self.arch = arch

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def train_step(self, x1, x2, xMotion):

        # get h representations, bolts resnet returns a list
        h1 = self.encoder(x1)[-1] #because resnet returns a list
        h2 = self.encoder(x2)[-1]
        hMotion = self.encoder(xMotion)[-1]

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        zMotion = self.projection(hMotion)
        
        return z1, z2, zMotion


    
    def forward(self, x):
        return self.encoder(x)[-1]
