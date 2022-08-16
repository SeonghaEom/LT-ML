CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size=448 --batch-size 32 --name=exp1 --seed=0 --model=resnet101 --finetune=$2 --wandb=fixed~0 --optim_config 4 $3
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size=448 --batch-size 32 --name=exp1 --seed=1 --model=resnet101 --finetune=$2 --wandb=fixed~0 --optim_config 4 $3
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size=448 --batch-size 32 --name=exp1 --seed=2 --model=resnet101 --finetune=$2 --wandb=fixed~0 --optim_config 4 $3

CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size=448 --batch-size 32 --name=exp1 --seed=0 --model=resnet101 --finetune=$2 --wandb=fixed~4 --optim_config 8 $3
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size=448 --batch-size 32 --name=exp1 --seed=1 --model=resnet101 --finetune=$2 --wandb=fixed~4 --optim_config 8 $3
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size=448 --batch-size 32 --name=exp1 --seed=2 --model=resnet101 --finetune=$2 --wandb=fixed~4 --optim_config 8 $3