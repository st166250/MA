# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from backbone.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys_Encoder
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys_Decoder

logger = logging.getLogger(__name__)


class SwinUnet_Encoder(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(SwinUnet_Encoder, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet_encoder = SwinTransformerSys_Encoder(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.swin_unet_encoder(x)
        return x


class SwinUnet_Decoder(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(SwinUnet_Decoder, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet_decoder = SwinTransformerSys_Decoder(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x, x_downsample):
        x_rec, x_seg = self.swin_unet_decoder(x, x_downsample)
        return x_rec, x_seg

