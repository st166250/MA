# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.nn import functional as F

#from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from backbone.config_swin_transformer import get_config
from backbone.vision_transformer import SwinUnet_Encoder as SwinViT_Encoder
from backbone.vision_transformer import SwinUnet_Decoder as SwinViT_Decoder
from monai.utils import ensure_tuple_rep

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

class Projection2(nn.Module):
    def __init__(self, input_dim=49*768, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim*4),
            nn.BatchNorm1d(self.hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim*4, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
    
class SSLHead(nn.Module):
    """Almost same as SSL_SwinUNet/models/ssl_head.py
    """
    def __init__(self, dim=768, 
                        hidden_mlp: int = 2048, #512 for resnet18 #2048 fo resnet50
                        feat_dim: int = 128,
                        first_conv: bool = True,
                        maxpool1: bool = True,):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, 224)
        window_size = ensure_tuple_rep(7, 224)
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        config = get_config()
        self.swinViT_encoder = SwinViT_Encoder(config, img_size=224, num_classes=2)
        self.projection = Projection(input_dim=49*768, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

    # for no pooling model
    def forward(self, x):
        x_out = self.swinViT_encoder(x)
        B, w, c = x_out.shape
        x_rec = x_out.view(B,-1)
        return x_rec

    # #for mean pooling model
    # def forward(self, x):
    #     x_out = self.swinViT_encoder(x)
    #     x_rec = x_out.mean(dim=1)
    #     return x_rec
    
    # #for mean pooling model
    # def train_step(self, x1, x2, xMotion):

    #     # get h representations, bolts resnet returns a list
    #     h1 = self.swinViT_encoder(x1)
    #     print("h1.shape: {}".format(h1.shape))
    #     h1 = h1.mean(dim=1)
    #     #B, w, c = h1.shape
    #     #h1 = h1.view(B,-1)
    #     print("h1.shape after view: {}".format(h1.shape))

    #     h2 = self.swinViT_encoder(x2)
    #     h2 = h2.mean(dim=1)

    #     hMotion = self.swinViT_encoder(xMotion)
    #     hMotion = hMotion.mean(dim=1)
    #     # get z representations
    #     z1 = self.projection(h1)
    #     z2 = self.projection(h2)
    #     zMotion = self.projection(hMotion)
        
    #     return z1, z2, zMotion

    #for no pooling
    def train_step(self, x1, x2, xMotion):

        # get h representations, bolts resnet returns a list
        h1 = self.swinViT_encoder(x1)
        B, w, c = h1.shape
        h1 = h1.view(B,-1)

        h2 = self.swinViT_encoder(x2)
        h2 = h2.view(B,-1)

        hMotion = self.swinViT_encoder(xMotion)
        hMotion = hMotion.view(B, -1)
        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        zMotion = self.projection(hMotion)
        
        return z1, z2, zMotion