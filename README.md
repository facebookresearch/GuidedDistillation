# Guided Distillation for Semi-Supervised Instance Segmentation
<!-- # Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022) -->

Tariq Berrada, Camille Couprie, Karteek Alahari, Jakob Verbeek.

[[`WACV website`](https://openaccess.thecvf.com/content/WACV2024/html/Berrada_Guided_Distillation_for_Semi-Supervised_Instance_Segmentation_WACV_2024_paper.html)][[`arXiv`](https://arxiv.org/abs/2308.02668)] [[`BibTeX`](#citing)]

<div align="center">
  <img src="illustrations/coco_illustration.png" width="100%" height="100%"/>
</div><br/>

[Guided Distillation](https://github.com/facebookresearch/GuidedDistillation) is a semi-supervised training methodology for instance segmentation building on the [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main) model.
It achieves substantial improvements with respect to the previous state-of-the-art in terms of mask-AP.
Most notably, our method outperforms the fully supervised Mask-RCNN COCO baseline while using only 2\% of the annotations.
Using a ViT (DinoV2) backbone, our method achieves 31.0 mask-AP while using 0.4\% of annotations only.

Our implementation is based on [detectron2](https://github.com/facebookresearch/detectron2) and provides support for both COCO and Cityscapes with multiple backbones such as R50, Swin, ViT (DETR) and ViT (DinoV2).

## Features
* Semi-supervised distillation training for instance segmentation with different percentages of labeled data.
* Tested on both COCO and Cityscapes.
* Support for R50, Swin, ViT (DETR) and ViT (DinoV2) backbones.

## Installation

Our codebase is based on detectron2 and Mask2Former.
An example of environment installing useful dependencies is provided in [install.md](install.md).

## Prepare Datasets for Mask2Former

For experiments with different amounts of labeled data, you will need to generate annotation structures for each of the percentages you want to use in your experiments, please follow the instructions at `tools/datasets` for this.

## Model Training 

Training is split into two consecutive steps.

* **Pre-training** : Train the model using the available labeled data.
* **Burn-in and distillation** : Train the student model using both labeled and unlabeled samples with targets provided by the teacher.

The following section provides examples of scripts to launch for different use cases.

### Example on Cityscapes with a R50 backbone

Example with R50 for cityscapes with 5\% of labeled data on 2 GPUs.

* Train teacher model with available labeled data only :
    ```
    python3 train_net.py --config-file ./configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml --num-gpus 2 --num-machines 1 SSL.PERCENTAGE 5 SSL.TRAIN_SSL False OUTPUT_DIR *OUTPUT/DIR/TEACHER*
    ```

* Train semi-supervised model using pretrained checkpoint
    ```
    python3 train_net.py --config-file ./configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml --num-gpus 2 --num-machines 1 SSL.PERCENTAGE 5 SSL.TRAIN_SSL True SSL.TEACHER_CKPT *PATH/TO/CKPT* OUTPUT_DIR *OUTPUT/DIR/STUDENT* SSL.BURNIN_ITER *NB_ITER*
    ```

For Swin backbones, pretrained weights are expected to be downloaded and converted according to Mask2Former's [scripts]([Title](https://github.com/facebookresearch/MaskFormer/tree/main/tools)).

### Example on COCO with a DINO backbone

Example with DINOv2 backbones, 0.4\% of labeled data (for percentage below 1\%, the argument takes 100/percentage=250 in this case): 

* Train teacher model with available labeled data only :
    ```
    python3 -W ignore train_net.py --config-file ./configs/coco/instance-segmentation/dinov2/maskformer2_dinov2_large_bs16_50ep.yaml --num-gpus 2 --num-machines 1 SSL.PERCENTAGE 250 SSL.TRAIN_SSL False OUTPUT_DIR *OUTPUT/DIR/TEACHER*
    ```

* Train semi-supervised model using pretrained checkpoint
    ```
    python3 -W ignore train_net.py --config-file ./configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml --num-gpus 2 --num-machines 1 SSL.PERCENTAGE 5 SSL.TRAIN_SSL True SSL.TEACHER_CKPT *PATH/TO/CKPT* OUTPUT_DIR *OUTPUT/DIR/STUDENT* SSL.BURNIN_ITER *NB_ITER*
    ```

You can choose to evaluate either the teacher, the student or both models during semi-supervised training. 
To this end you can add the `SSL.EVAL_WHO` argument to your script and set it to either `STUDENT` (default), `TEACHER` or `BOTH`.

## <a name="citing"></a>Citing Guided Distillation

If you use Guided Distillation in your research or wish to refer to the baseline results published in the manuscript, please use the following BibTeX entry.

```BibTeX
@InProceedings{Berrada_2024_WACV,
    author    = {Berrada, Tariq and Couprie, Camille and Alahari, Karteek and Verbeek, Jakob},
    title     = {Guided Distillation for Semi-Supervised Instance Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {475-483}
}
```

## Acknowledgement

Code is largely based on [MaskFormer](https://github.com/facebookresearch/MaskFormer) and [Detectron2](https://github.com/facebookresearch/detectron2).

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
"*Guided Distillation for Semi-Supervised Instance Segmentation*" is <YOUR LICENSE HERE> licensed, as found in the LICENSE file.
