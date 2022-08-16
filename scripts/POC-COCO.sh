CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --base --wandb=fixed_fc --optim_config 2 $2
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --base --wandb=l1_fc --optim_config 1 2 $2
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --base --wandb=all_fc --optim_config 0 2 $2
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --transformer --wandb=fixed_SA --optim_config 8 $2
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --transformer --wandb=l1_SA --optim_config 7 8 $2
CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --transformer --wandb=all_SA --optim_config 11 8 $2