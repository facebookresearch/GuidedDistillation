# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, json, tqdm
import numpy as np
from argparse import ArgumentParser
from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='Cityscapes data processing.',
        description='Create json files for unlabeled dataset.',
    )

    parser.add_argument('--source_img_folder', type=str, required=True)
    parser.add_argument('--dest_img_folder', type=str, required=True)

    args = parser.parse_args()
    
    new_root = os.path.join(args.dest_img_folder, 'leftImg8bit', 'train')
    root = args.source_img_folder
    os.makedirs(new_root, exist_ok=True)

    new = {'annotations': [],
      'categories': [],
      'images': []}

    for city in tqdm.tqdm(os.listdir(root)):  
        #os.makedirs(new_root+'/'+city, exist_ok=True)
        for x in os.listdir(root+'/'+city):
            if x.endswith('.png'):
                img_ = Image.open(os.path.join(root, city, x))
                os.symlink(os.path.join(root, city, x), os.path.join(new_root, x))
                width, height = img_.size
                id_ = x.split('.')[0].replace('_leftImg8bit', '') #'_'.join(x.split('_')[:3])
                img_info = {
                    'file_name': x,
                    'id': id_,
                    'height': height,
                    'width': width,
                }
                annot_info = {
                    'file_name': '',
                    'image_id': id_,
                    'segments_info': [],
                }
                # print(img_info)
                new['images'].append(img_info)
                new['annotations'].append(annot_info)

    with open(os.path.join(args.dest_img_folder, 'cityscapes_unlabel_train.json'), 'w') as f: 
        json.dump(new, f)