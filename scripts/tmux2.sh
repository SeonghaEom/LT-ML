
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=resnet101 --finetune=$2  --optim_config 8 --wandb=freeze $3 $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=resnet101 --finetune=$2  --optim_config 8 --wandb=freeze $3 $4 $5

CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=resnet101 --finetune=$2  --optim_config 4 --wandb=finetune $3 $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=resnet101 --finetune=$2  --optim_config 4 --wandb=finetune $3 $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=resnet101 --finetune=$2  --optim_config 4 --wandb=finetune $3 $4 $5