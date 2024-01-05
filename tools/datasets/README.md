# Data Preprocessing

Our code works by reading image and annotation paths from json files corresponding to different data fractions. The scripts in this folder can be used to generate random splits and writing them to the json files.

For every dataset, there is a labeled and an unlabled dataset. 

## Unlabeled datasets
The unlabeled dataset has the following structure :

```
{dataset}_unlabel/
    images/
    {dataset}_unlabel_train.json
```

and the annotation json has the same structure as the unlabeled json, with an empty 'annotations' tab :

```
root
    annotations:
        0: ""
            image_id: "{ID}"
            segments_info: []
        ...
    categories:
        []
    images:
        0:
            file_name: "{FILENAME}"
            id: "{ID}"
            height: {IM_HEIGHT}
            width: {IM_WIDTH}
        ...
```

To generate the json file, you should have your unlabeled images with ```{dataset}_unlabel/images``` then launch `generate_{dataset}_unl.py` to generate the json file. Detailed, step by step instructions are provided in the "Detailed instruction" section below.

## Labeled datasets

We generate json files for different annotation percentages. 

```
python generate_coco_splits.py --json_dir_file {annotations_dir}/instances_train2017.json --percentages 0.5 1 10 
```
Note: For percentages < 1%, the output filename is 100/percentage to be an integer, e.g. 200 for 0.5%.

For Cityscapes, the default dataloader does not read a global json file but individual annotations from the gtFine folder, therefore we make different copies of the annotations directory each with a different percentage of labeled data. 

```
python generate_cityscapes_splits.py --dataset_dir {} --percentage 5
```

The cityscapes dataset directory should then have the following struture :

```
cityscapes/
    leftImg8bit/
    gtFine/
    gtFine_5/
    gtFine_10/
    gtFine_20/
```

# Detailed Instructions 

run
```
DATA_PATH="/path_to/COCODataset/"
GD_PATH="/path_to/GuidedDistillation/"
```

## Coco Unlabeled dataset


```
cd $GD_PATH/tools/datasets;
mkdir data_for_Guided_Distillation; 
cd data_for_Guided_Distillation; mkdir coco_unlabel; cd coco_unlabel
ln -s $DATA_PATH/val2017 val2017
ln -s $DATA_PATH/train2017 images
```
Then, you can generate json annotations:
```
cd ../..;
python generate_coco_unl.py --img_folder data_for_Guided_Distillation/coco_unlabel
```

## Coco Labeled dataset

```
cd $GD_PATH/tools/datasets/data_for_Guided_Distillation/; mkdir coco; cd coco
ln -s $DATA_PATH/train2017 train2017
mkdir annotations;
cp $DATA_PATH/annotations/instances_train2017.json annotations/
cp $DATA_PATH/annotations/instances_val2017.json annotations/
```

Then you can create the json annotations:
```
cd ../..;
python generate_coco_splits.py --json_dir data_for_Guided_Distillation/coco/annotations/instances_train2017.json -percentages 0.1 1 5 10
```

Finally:
```
export DETECTRON2_DATASETS=$GD_PATH/tools/datasets/data_for_Guided_Distillation
```

## Cityscapes

Create the unlabeled json:
```
cd data_for_Guided_Distillation; mkdir cityscapes_unlabel; cd cityscapes_unlabel
cd tools/datasets;
python generate_cityscapes_unl.py --dest_img_folder your_path/GuidedDistillation/data_for_Guided_Distillation/cityscapes_unlabel --source_img_folder /datasets01/cityscapes/112817/leftImg8bit/train
```

Create the annotation files:
```
cd yourpath/data_for_Guided_Distillation/; mkdir cityscapes; cd cityscapes
cd tools/datasets;
python generate_cityscapes_splits.py --dataset_dir /datasets01/cityscapes/112817 --percentage 5 --dest_dir your_path/data_for_Guided_Distillation/cityscapes/
```

