CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~0 --optim_config 4 5 6 7 8 $2 --name=exp2 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~0 --optim_config 4 5 6 7 8 $2 --name=exp2 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~0 --optim_config 4 5 6 7 8 $2 --name=exp2 --seed=2


CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~1 --optim_config 5 6 7 8 $2 --name=exp2 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~1 --optim_config 5 6 7 8 $2 --name=exp2 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~1 --optim_config 5 6 7 8 $2 --name=exp2 --seed=2


CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~2 --optim_config 6 7 8 $2 --name=exp2 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~2 --optim_config 6 7 8 $2 --name=exp2 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~2 --optim_config 6 7 8 $2 --name=exp2 --seed=2


CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~3 --optim_config 7 8 $2 --name=exp2 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~3 --optim_config 7 8 $2 --name=exp2 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~3 --optim_config 7 8 $2 --name=exp2 --seed=2


CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~4 --optim_config 8 $2 --name=exp2 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~4 --optim_config 8 $2 --name=exp2 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --image-size 448 --batch-size $4 --base --wandb=fixed~4 --optim_config 8 $2 --name=exp2 --seed=2
