# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, json, tqdm
import numpy as np
from argparse import ArgumentParser

import shutil

def read_split(filepath):
    files = open(filepath, "r")
    return [u[:-1] for u in files] 

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='Cityscapes data processing.',
        description='Create json files for different labeled fractions',
    )

    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--split_dir', type=str, default="./data_splits/cityscapes/")
    parser.add_argument('--percentage', type=int,)

    args = parser.parse_args()

    root = os.path.join(args.dataset_dir, 'gtFine', 'train')
    root_img = os.path.join(args.dataset_dir, 'leftImg8bit', 'train') 

    if args.percentage:
        percentages = [args.percentage]
    else:
        percentages = [5, 10, 20, 30]

    # Create symlink for 100\% percentage.
    new_root = os.path.join(args.dest_dir, 'gtFine', f'train')
    new_root_img = os.path.join(args.dest_dir, 'leftImg8bit', f'train')
    replace_k = [
        ("_polygons.json", "_color.png"),
        ("_polygons.json", "_instanceIds.png"),
        ("_polygons.json", "_labelIds.png"),
        ("_polygons.json", "_labelTrainIds.png")
    ]
    os.makedirs(new_root, exist_ok=True)
    os.makedirs(new_root_img, exist_ok=True)
    for city in tqdm.tqdm(os.listdir(root)):
        os.makedirs(os.path.join(new_root, city), exist_ok=True)
        files = [u for u in os.listdir(os.path.join(root, city)) if u.endswith('.json')]
        for f_ in files:
            os.symlink(os.path.join(root, city, f_), os.path.join(new_root, city, f_))
            for k, v in replace_k:
                os.symlink(os.path.join(root, city, f_.replace(k, v)), os.path.join(new_root, city, f_.replace(k, v)))
            os.makedirs(os.path.join(new_root_img, city), exist_ok=True)
            os.symlink(os.path.join(root_img, city, f_.replace("_gtFine_polygons.json", "_leftImg8bit.png")), os.path.join(new_root_img, city, f_.replace("_gtFine_polygons.json", "_leftImg8bit.png")))

    # Same but for validation set.
    root_val = os.path.join(args.dataset_dir, 'gtFine', 'val')
    root_img_val = os.path.join(args.dataset_dir, 'leftImg8bit', 'val') 
    new_root_val = os.path.join(args.dest_dir, 'gtFine', f'val')
    new_root_img_val = os.path.join(args.dest_dir, 'leftImg8bit', f'val')
    replace_k = [
        ("_polygons.json", "_color.png"),
        ("_polygons.json", "_instanceIds.png"),
        ("_polygons.json", "_labelIds.png"),
        ("_polygons.json", "_labelTrainIds.png")
    ]
    os.makedirs(new_root_val, exist_ok=True)
    os.makedirs(new_root_img_val, exist_ok=True)
    for city in tqdm.tqdm(os.listdir(root_val)):
        os.makedirs(os.path.join(new_root_val, city), exist_ok=True)
        files = [u for u in os.listdir(os.path.join(root_val, city)) if u.endswith('.json')]
        for f_ in files:
            os.symlink(os.path.join(root_val, city, f_), os.path.join(new_root_val, city, f_))
            for k, v in replace_k:
                os.symlink(os.path.join(root_val, city, f_.replace(k, v)), os.path.join(new_root_val, city, f_.replace(k, v)))
            os.makedirs(os.path.join(new_root_img_val, city), exist_ok=True)
            os.symlink(os.path.join(root_img_val, city, f_.replace("_gtFine_polygons.json", "_leftImg8bit.png")), os.path.join(new_root_img_val, city, f_.replace("_gtFine_polygons.json", "_leftImg8bit.png")))

    for perc in percentages:
        print(perc)
        new_root = os.path.join(args.dest_dir, 'gtFine', f'train_{perc}')
        new_root_img = os.path.join(args.dest_dir, 'leftImg8bit', f'train_{perc}')
        replace_k = [
            ("_instanceIds.png", "_color.png"),
            ("_instanceIds.png", "_polygons.json"),
            ("_instanceIds.png", "_labelIds.png"),
            ("_instanceIds.png", "_labelTrainIds.png")
        ]
        os.makedirs(new_root, exist_ok=True)
        os.makedirs(new_root_img, exist_ok=True)
        for city in tqdm.tqdm(os.listdir(root)):
            os.makedirs(os.path.join(new_root, city), exist_ok=True)
        # files = [u for u in os.listdir(os.path.join(root, city)) if u.endswith('.json')]
        keep = read_split(os.path.join(args.split_dir, "perc_"+str(perc) + ".txt"))
        # keep = np.random.choice(files, size=int(perc*len(files)/100), replace=False)
        for f_ in keep:
            city = f_.split("_")[0]
            os.symlink(os.path.join(root, city, f_), os.path.join(new_root, city, f_))
            for k, v in replace_k:
                os.symlink(os.path.join(root, city, f_.replace(k, v)), os.path.join(new_root, city, f_.replace(k, v)))
            os.makedirs(os.path.join(new_root_img, city), exist_ok=True)
            os.symlink(os.path.join(root_img, city, f_.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")), os.path.join(new_root_img, city, f_.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")))