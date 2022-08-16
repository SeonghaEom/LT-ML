# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=$2 --finetune=base --wandb=fixed~$3 --optim_config $4 $5
# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=$2 --finetune=base --wandb=fixed~$3 --optim_config $4 $5
# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=$2 --finetune=base --wandb=fixed~$3 --optim_config $4 $5

CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=$2 --finetune=gcn --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=$2 --finetune=gcn --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=$2 --finetune=gcn --wandb=fixed~$3 --optim_config $4 $5

# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=$2 --finetune=sa --wandb=fixed~$3 --optim_config $4 $5
# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=$2 --finetune=sa --wandb=fixed~$3 --optim_config $4 $5
# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=$2 --finetune=sa --wandb=fixed~$3 --optim_config $4 $5

CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=$2 --finetune=transformer_encoder --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=$2 --finetune=transformer_encoder --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=$2 --finetune=transformer_encoder --wandb=fixed~$3 --optim_config $4 $5

CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=$2 --finetune=se --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=$2 --finetune=se --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=$2 --finetune=se --wandb=fixed~$3 --optim_config $4 $5

CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=0 --model=$2 --finetune=mha --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=1 --model=$2 --finetune=mha --wandb=fixed~$3 --optim_config $4 $5
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 32 --name=exp1 --seed=2 --model=$2 --finetune=mha --wandb=fixed~$3 --optim_config $4 $5