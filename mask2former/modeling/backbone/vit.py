# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from pytorch_pretrained_vit import ViT

@BACKBONE_REGISTRY.register()
class ViTBaseBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.model = ViT('B_16', pretrained=True, image_size=1024).train().cuda() # 'B_16_imagenet1k' cfg.INPUT.CROP.SIZE[0]
        del self.model.fc
        del self.model.norm
        self.qkv_out = None
        self.token_size = 16
        self.factors = {
            'res2': 4,
            'res3': 8,
            'res4':16,
            'res5': 32,
        }
        self.base=128
        # self.w, self.h = input_shape
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
        # self.model.blocks[11].attn.qkv.register_forward_hook(self.extract_hook())

        self.convs = nn.ModuleList([nn.Conv2d(768, self.base*fact//4, kernel_size=1) for fact in self.factors.values()])
        
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 128,
            "res3": 256,
            "res4": 512,
            "res5": 1024,
        }
    # def extract_hook(self):
    #     def hook(module, input, output):
    #         self.qkv_out = output
    #     return hook

    def get_divisible_size(self, w, h):
        return w - w%16, h - h%16

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 14

    def forward(self, x):
        w, h = x.shape[-2:]
        dw, dh = self.get_divisible_size(w, h)
        x_inp = F.interpolate(x, size=(dw, dh))
        pw, ph = dw//self.token_size, dh//self.token_size
        print('IN ------------- ', x_inp.shape)
        feat = self.model(x_inp)
        print('out ---------- ', feat.shape)
        # cls_token = y[:, :1, :]
        patch = feat[:, 1:, :]

        patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

        feat_dict = {}
        for (k, scale), conv in zip(self.factors.items(), self.convs):
            new_patch = F.interpolate(patch, size=(w//scale, h//scale))
            feat_dict[k] = conv(new_patch)

        return feat_dict


@BACKBONE_REGISTRY.register()
class ViTLargeBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.model = ViT('L_16', pretrained=True, image_size=cfg.INPUT.CROP.SIZE[0]).cuda()
        del model.fc
        self.qkv_out = None
        self.token_size = 16
        self.factors = {
            'res2': 4,
            'res3': 8,
            'res4':16,
            'res5': 32,
        }
        self.base=128
        # self.w, self.h = input_shape
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
        # self.model.blocks[11].attn.qkv.register_forward_hook(self.extract_hook())
        self.feature_size = 1024
        self.convs = nn.ModuleList([nn.Conv2d(self.feature_size, self.base*fact//4, kernel_size=1) for fact in self.factors.values()])
        
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 128,
            "res3": 256,
            "res4": 512,
            "res5": 1024,
        }
    # def extract_hook(self):
    #     def hook(module, input, output):
    #         self.qkv_out = output
    #     return hook

    def get_divisible_size(self, w, h):
        return w + (16 - w%16), h+ (16 - h%16)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 16

    def forward(self, x):
        w, h = x.shape[-2:]
        dw, dh = self.get_divisible_size(w, h)
        x_inp = F.interpolate(x, size=(dw, dh))
        pw, ph = dw//self.token_size, dh//self.token_size
        
        feat = self.model(x_inp)
        # cls_token = y[:, :1, :]
        patch = feat[:, 1:, :]

        patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

        feat_dict = {}
        for (k, scale), conv in zip(self.factors.items(), self.convs):
            new_patch = F.interpolate(patch, size=(w//scale, h//scale))
            feat_dict[k] = conv(new_patch)

        return feat_dict