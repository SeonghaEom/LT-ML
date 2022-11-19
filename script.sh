## arguments: gpu, datast[coco, voc], batch-size, intermediate representation extracted at block of index[0, 1, 2]
#### intermediate representation extracted right after the first block
CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/$2 --dataset=$2 --image-size 448 --batch-size $3  --seed=0 --model=resnet101 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=$4
# CUDA_VISIBLE_DEVICES=$1 python3 TL.py data/$2 --dataset=$2 --image-size 384 --batch-size $3  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=$4


## EXAMPLES for Table1 and Table2a
#### resnet101 + intermediate representation extracted right after the second block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 64  --seed=0 --model=resnet101 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

#### resnet50 + intermediate representation extracted right after the third block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 448 --batch-size 64  --seed=0 --model=resnet50 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=2

#### swin + intermediate representation extracted right after the first block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=0 

#### swin large + intermediate representation extracted right after the second block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

#### finetune
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --finetune

#### Linear probe (baseline)
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=1 --epochs=30  --lr_scheduler


## EXAMPLES for Table2b Pascal-VOC
#### resnet101 + intermediate representation extracted right after the second block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 448 --batch-size 64  --seed=0 --model=resnet101 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

#### resnet50 + intermediate representation extracted right after the third block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 448 --batch-size 64  --seed=0 --model=resnet50 --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=2

#### swin + intermediate representation extracted right after the first block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=0 

#### swin large + intermediate representation extracted right after the second block
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --intermediate --where=1

#### finetune
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=0 --epochs=30  --lr_scheduler --finetune

#### Linear probe (baseline)
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin_large --optim_config=1 --epochs=30  --lr_scheduler


## EXAMPLES for Appendix (Proposed method w/ fine-tune)
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=resnet101 --optim_config=0 --epochs=30  --lr_scheduler --finetune --intermediate --where=1
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/voc --dataset=voc --image-size 384 --batch-size 50  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --finetune --intermediate --where=1
# CUDA_VISIBLE_DEVICES=0 python3 TL.py data/coco --dataset=coco --image-size 384 --batch-size 50  --seed=0 --model=swin --optim_config=0 --epochs=30  --lr_scheduler --finetune --intermediate --where=1