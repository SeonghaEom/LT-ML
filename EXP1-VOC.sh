CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_gcn --optim_config 6 7 8 $2 --name=exp1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_gcn --optim_config 6 7 8 $2 --name=exp1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_gcn --optim_config 6 7 8 $2 --name=exp1 --seed=2

CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_fc --base --optim_config 6 7 8 $2 --name=exp1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_fc --base --optim_config 6 7 8 $2 --name=exp1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_fc --base --optim_config 6 7 8 $2 --name=exp1 --seed=2

CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_SA --transformer --optim_config 6 7 8 $2 --name=exp1 --seed=0
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_SA --transformer --optim_config 6 7 8 $2 --name=exp1 --seed=1
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --wandb=l2_SA --transformer --optim_config 6 7 8 $2 --name=exp1 --seed=2
