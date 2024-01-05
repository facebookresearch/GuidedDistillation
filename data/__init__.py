# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.data import transforms  # isort:skip

from .build import (
    build_batch_data_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    load_proposals_into_dataset,
    print_instances_class_histogram,
)
from .catalog import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.data.common import DatasetFromList, MapDataset, ToIterableDataset
from detectron2.data.dataset_mapper import DatasetMapper

# ensure the builtin datasets are registered
from . import datasets #, samplers  # isort:skip

from detectron2.data import samplers
__all__ = [k for k in globals().keys() if not k.startswith("_")]
