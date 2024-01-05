# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, json, tqdm
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='COCO data processing.',
        description='Create json files for different labeled fractions from instances_train2017.json',
    )

    parser.add_argument('--json_dir_file', type=str, required=True)
    parser.add_argument('-percentages', nargs="+", type=float) 

    args = parser.parse_args()

    data = json.load(open(args.json_dir_file, "r"))
    print('Parsed JSON :')

    for k, v in data.items():
        print(k, len(v))
        
    if args.percentages:
        percentages = list(set(args.percentages)) 
    else: 
        percentages = [0.4, 1, 2, 5, 10, 30]
        
    for perc in tqdm.tqdm(percentages):
        
        nb_im = int(perc*len(data['images'])/100)
        print("percentage ", perc, "nb_im ", nb_im)
        if perc < 1:
            perc = int(100/perc)
        else:
            perc = int(perc)
            
        new_images = np.random.choice(data['images'], size=nb_im, replace=False)
        new_ids = [x['id'] for x in new_images]
        new_annotations = [ann for ann in data['annotations'] if ann['image_id'] in new_ids]

        new_data = {
            'annotations': new_annotations,
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': data['categories'],
            'images': list(new_images),
        }
        new_dir = args.json_dir_file.replace('instances_train2017.json', f'instances_train2017_{perc}.json')
       
        assert new_dir != args.json_dir_file
        json.dump(new_data, open(new_dir, 'w'))
    
