for i in 1 2 3 4
do
  for j in 1 2 4 8 16
  do
    CUDA_VISIBLE_DEVICES=$1 python3 demo_coco_mllt.py data/coco --LT --image-size 448 --batch-size 32 --name=exp3 --seed=$4 --model=$2 --finetune=transformer_encoder --wandb=hi --optim_config $3 --num_block=$i --num_head=$j --epochs=10
  done
done

