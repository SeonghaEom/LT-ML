# Layover Intermediate Layer for Multi-Label Classification in Efficient Transfer Learning (HITY Workshop NeurIPS 2022)


Official Pytorch implementation of ****[Layover Intermediate Layer for Multi-Label Classification in Efficient Transfer Learning](https://openreview.net/forum?id=mbOHmKLxBH)****

## Setup


This setting requires CUDA 11. However you can still use your own environment by installing requirements including Pytorch and Torchvision.

1. Install conda environment and activate it.

```bash
conda env create -f env.yaml
conda activate LTML
```

2. Datasets will be downloaded automatically. (VOC takes few minutes, COCO takes 1~2 hours)
    1. MS-COCO 2017 [[link](https://cocodataset.org/#home)]
    2. Pascal VOC 2007 [[link](http://host.robots.ox.ac.uk/pascal/VOC/)]

## MS-COCO
There are more examples in script.sh to reproduce results

```bash
## EXAMPLES for Table1 and Table2a
### resnet101 + intermediate representation extracted right after the second block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 64  --seed=0 --model=resnet101 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

### resnet50 + intermediate representation extracted right after the third block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 64  --seed=0 --model=resnet50 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=2

### swin + intermediate representation extracted right after the first block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=0 

### swin large + intermediate representation extracted right after the second block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

### finetune
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --finetune

### Linear probe (baseline)
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=1 --epochs=30  --lr_scheduler
```

## Pascal-VOC
There are more examples in script.sh to reproduce results


```bash
## EXAMPLES for Table2b Pascal-VOC
### resnet101 + intermediate representation extracted right after the second block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 448 --batch-size 64  --seed=0 --model=resnet101 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

### resnet50 + intermediate representation extracted right after the third block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 448 --batch-size 64  --seed=0 --model=resnet50 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=2

### swin + intermediate representation extracted right after the first block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=0 

### swin large + intermediate representation extracted right after the second block
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

### finetune
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --finetune

### Linear probe (baseline)
CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=1 --epochs=30  --lr_scheduler
```
