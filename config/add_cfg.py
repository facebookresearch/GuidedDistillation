# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config import CfgNode as CN

def add_ssl_config(cfg):
    # ---------------------------------------------------------------------------- #
    # SSL options
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.SSL = CN()
    _C.SSL.TRAIN_SSL = True
    _C.SSL.TEACHER_CKPT = ""
    _C.SSL.BURNIN_ITER = 50000
    _C.SSL.PERCENTAGE = 100
    _C.SSL.FREQ = 1 #3
    _C.SSL.EMA_DECAY = 0.9996
    _C.SSL.CKPT_TARGET = 'TEACHER'
    _C.SSL.EVAL_WHO = "STUDENT"
    _C.SSL.WEIGHTS = ""