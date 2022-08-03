CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_gcn --optim_config 5 6 7 8 $2 --name=exp1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_gcn --optim_config 5 6 7 8 $2 --name=exp1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_gcn --optim_config 5 6 7 8 $2 --name=exp1 --seed=2

CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_fc --base --optim_config 5 6 7 8 $2 --name=exp1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_fc --base --optim_config 5 6 7 8 $2 --name=exp1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_fc --base --optim_config 5 6 7 8 $2 --name=exp1 --seed=2

CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_SA --transformer --optim_config 5 6 7 8 $2 --name=exp1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_SA --transformer --optim_config 5 6 7 8 $2 --name=exp1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --wandb=l3_SA --transformer --optim_config 5 6 7 8 $2 --name=exp1 --seed=2
