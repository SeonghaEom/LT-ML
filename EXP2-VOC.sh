CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~0 --optim_config 4 5 6 7 8 $2 --name=exp2 --seed=0 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~0 --optim_config 4 5 6 7 8 $2 --name=exp2 --seed=1 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~0 --optim_config 4 5 6 7 8 $2 --name=exp2 --seed=2 --model=$3


CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~1 --optim_config 5 6 7 8 $2 --name=exp2 --seed=0 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~1 --optim_config 5 6 7 8 $2 --name=exp2 --seed=1 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~1 --optim_config 5 6 7 8 $2 --name=exp2 --seed=2 --model=$3


CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~2 --optim_config 6 7 8 $2 --name=exp2 --seed=0 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~2 --optim_config 6 7 8 $2 --name=exp2 --seed=1 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~2 --optim_config 6 7 8 $2 --name=exp2 --seed=2 --model=$3


CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~3 --optim_config 7 8 $2 --name=exp2 --seed=0 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~3 --optim_config 7 8 $2 --name=exp2 --seed=1 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~3 --optim_config 7 8 $2 --name=exp2 --seed=2 --model=$3


CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~4 --optim_config 8 $2 --name=exp2 --seed=0 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~4 --optim_config 8 $2 --name=exp2 --seed=1 --model=$3
CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_mllt.py data/voc --image-size 448 --batch-size $4 --wandb=fixed~4 --optim_config 8 $2 --name=exp2 --seed=2 --model=$3
