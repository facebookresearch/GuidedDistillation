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
# from pytorch_pretrained_vit import ViT
import torch
import torch.nn.functional as F
import torch.nn as nn

@BACKBONE_REGISTRY.register()
# import math

class DeiTBaseBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
       
        self.model = torch.hub.load('facebookresearch/deit', 'deit_base_patch16_384', pretrained=True).cuda()
        self.model.patch_embed.img_size = (1024, 1024)
        self.model.patch_size=16
       
        del self.model.head
        self.overwrite_pos_encoding()
        
        self.qkv_out = None
        self.token_size = 16
        self.factors = {
            'res2': 4,
            'res3': 8,
            'res4':16,
            'res5': 32,
        }
        self.base=128
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES

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
        return 16

    def overwrite_pos_encoding(self):

        def interpolate_pos_encoding(x, w, h):
            dh, dw = 384, 384
            token_size=self.model.patch_size
            pw, ph = dw//token_size, dh//token_size
            patch = self.model.pos_embed[:, 1:, :]
            patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

            pw2, ph2 = w//token_size, h//token_size
            new_patch = F.interpolate(patch, size=(pw2, ph2), mode='bilinear')
            new_patch = new_patch.permute(0, 2, 3, 1).reshape(new_patch.shape[0], pw2*ph2, new_patch.shape[1])
            new_embed = torch.cat([self.model.pos_embed[:, :1, :], new_patch], dim=1)

            return new_embed

        # Add possiblity to interpolate positional encoing to the model.
        self.model.interpolate_pos_encoding = interpolate_pos_encoding

        def _pos_embed(x, w, h):
            if self.model.no_embed_class:
                # deit-3, updated JAX (big vision)
                # position embedding does not overlap with class token, add then concat
                x = x + self.model.interpolate_pos_encoding(self.model.pos_embed, w, h)
                if self.model.cls_token is not None:
                    x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            else:
                # original timm, JAX, and deit vit impl
                # pos_embed has entry for class token, concat then add
                if self.model.cls_token is not None:
                    x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + self.model.interpolate_pos_encoding(self.model.pos_embed, w, h)
            return self.model.pos_drop(x)
        setattr(self.model, "_pos_embed", _pos_embed)

    def forward_features(self, x):
        w, h = x.shape[-2:]
        self.model.patch_embed.img_size = (w, h)
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x, w, h)
        x = self.model.norm_pre(x)
        if self.model.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.model.blocks, x)
        else:
            x = self.model.blocks(x)
        x = self.model.norm(x)
        return x
    
    def forward(self, x):
        w, h = x.shape[-2:]
        dw, dh = self.get_divisible_size(w, h)
        x_inp = F.interpolate(x, size=(dw, dh))
        pw, ph = dw//self.token_size, dh//self.token_size
        feat = self.forward_features(x_inp)
        # cls_token = y[:, :1, :]
        patch = feat[:, 1:, :]

        patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

        feat_dict = {}
        for (k, scale), conv in zip(self.factors.items(), self.convs):
            new_patch = F.interpolate(patch, size=(w//scale, h//scale))
            feat_dict[k] = conv(new_patch)

        return feat_dict


# class DeiTBaseBackbone(Backbone):
#     def __init__(self, cfg, input_shape):
#         super().__init__()
#         # self.model = ViT('B_16', pretrained=True, image_size=1024).train().cuda() # 'B_16_imagenet1k' cfg.INPUT.CROP.SIZE[0]

#         self.model = torch.hub.load('facebookresearch/deit', 'deit_base_patch16_384', pretrained=True).cuda()
#         self.model.patch_embed.img_size = (1024, 1024)
#         # new_embed = F.interpolate(self.model.pos_embed.permute(0, 2, 1), size=(4097,)).permute(0, 2, 1)

#         dh, dw = 384, 384
#         token_size=16
#         pw, ph = dw//token_size, dh//token_size
#         patch = self.model.pos_embed[:, 1:, :]
#         patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

#         pw2, ph2 = 1024//token_size, 1024//token_size
#         new_patch = F.interpolate(patch, size=(pw2, ph2))
#         new_patch = new_patch.permute(0, 2, 3, 1).reshape(new_patch.shape[0], pw2*ph2, new_patch.shape[1])
#         new_embed = torch.cat([self.model.pos_embed[:, :1, :], new_patch], dim=1)

#         self.model.pos_embed = nn.Parameter(new_embed.to(self.model.pos_embed))
#         del self.model.head
        
#         self.qkv_out = None
#         self.token_size = 16
#         self.factors = {
#             'res2': 4,
#             'res3': 8,
#             'res4':16,
#             'res5': 32,
#         }
#         self.base=128
#         # self.w, self.h = input_shape
#         self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
#         # self.model.blocks[11].attn.qkv.register_forward_hook(self.extract_hook())

#         self.convs = nn.ModuleList([nn.Conv2d(768, self.base*fact//4, kernel_size=1) for fact in self.factors.values()])
        
#         self._out_feature_strides = {
#             "res2": 4,
#             "res3": 8,
#             "res4": 16,
#             "res5": 32,
#         }
#         self._out_feature_channels = {
#             "res2": 128,
#             "res3": 256,
#             "res4": 512,
#             "res5": 1024,
#         }
#     # def extract_hook(self):
#     #     def hook(module, input, output):
#     #         self.qkv_out = output
#     #     return hook

#     def get_divisible_size(self, w, h):
#         return w - w%16, h - h%16

#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }

#     @property
#     def size_divisibility(self):
#         return 16

#     def forward(self, x):
#         w, h = x.shape[-2:]
#         dw, dh = (1024, 1024) # self.get_divisible_size(w, h)
#         x_inp = F.interpolate(x, size=(dw, dh))
#         pw, ph = dw//self.token_size, dh//self.token_size
#         # print('IN ------------- ', x_inp.shape)
#         feat = self.model.forward_features(x_inp)
#         # print('out ---------- ', feat.shape)
#         # cls_token = y[:, :1, :]
#         patch = feat[:, 1:, :]

#         patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

#         feat_dict = {}
#         for (k, scale), conv in zip(self.factors.items(), self.convs):
#             new_patch = F.interpolate(patch, size=(w//scale, h//scale))
#             feat_dict[k] = conv(new_patch)

#         return feat_dict





# @BACKBONE_REGISTRY.register()
# class ViTLargeBackbone(Backbone):
#     def __init__(self, cfg, input_shape):
#         super().__init__()
#         self.model = ViT('L_16', pretrained=True, image_size=cfg.INPUT.CROP.SIZE[0]).cuda()
#         del model.fc
#         self.qkv_out = None
#         self.token_size = 16
#         self.factors = {
#             'res2': 4,
#             'res3': 8,
#             'res4':16,
#             'res5': 32,
#         }
#         self.base=128
#         # self.w, self.h = input_shape
#         self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
#         # self.model.blocks[11].attn.qkv.register_forward_hook(self.extract_hook())
#         self.feature_size = 1024
#         self.convs = nn.ModuleList([nn.Conv2d(self.feature_size, self.base*fact//4, kernel_size=1) for fact in self.factors.values()])
        
#         self._out_feature_strides = {
#             "res2": 4,
#             "res3": 8,
#             "res4": 16,
#             "res5": 32,
#         }
#         self._out_feature_channels = {
#             "res2": 128,
#             "res3": 256,
#             "res4": 512,
#             "res5": 1024,
#         }
#     # def extract_hook(self):
#     #     def hook(module, input, output):
#     #         self.qkv_out = output
#     #     return hook

#     def get_divisible_size(self, w, h):
#         return w + (16 - w%16), h+ (16 - h%16)

#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }

#     @property
#     def size_divisibility(self):
#         return 16

#     def forward(self, x):
#         w, h = x.shape[-2:]
#         dw, dh = self.get_divisible_size(w, h)
#         x_inp = F.interpolate(x, size=(dw, dh))
#         pw, ph = dw//self.token_size, dh//self.token_size
        
#         feat = self.model(x_inp)
#         # cls_token = y[:, :1, :]
#         patch = feat[:, 1:, :]

#         patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(0, 3, 1, 2)

#         feat_dict = {}
#         for (k, scale), conv in zip(self.factors.items(), self.convs):
#             new_patch = F.interpolate(patch, size=(w//scale, h//scale))
#             feat_dict[k] = conv(new_patch)

#         return feat_dict