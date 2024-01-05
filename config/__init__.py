# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config.compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable
from detectron2.config.instantiate import instantiate
from detectron2.config.lazy import LazyCall, LazyConfig

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "downgrade_config",
    "upgrade_config",
    "configurable",
    "instantiate",
    "LazyCall",
    "LazyConfig",
]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
