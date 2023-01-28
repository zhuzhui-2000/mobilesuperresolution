#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Experiments

model_type='basic_mv'  # NAS_MODEL / BASIC_MODEL

epochs=50
num_patches=3   # default 1000
train_batch_size=16 # default 16
lr_patch_size=64    # default 48

# arch
scale=4
num_blocks=16 #16
num_residual_units=24

num_gpus=$(awk -F '[0-9]' '{print NF-1}' <<<"$CUDA_VISIBLE_DEVICES")
echo Using $num_gpus GPUs

now=$(date +'%b%d_%H_%M_%S')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=wdsr_b_x${scale}_${num_blocks}_${num_residual_units}_${now}
else
  job_dir=${experiment_name}_${now}
fi

printf '%s\n' "Job save in runs/$job_dir"

if [ -d "runs/$job_dir" ]; then
  printf '%s\n' "Removing runs/$job_dir"
  rm -rf "runs/$job_dir"
fi

############ ACTIVATE environment here  ##############
#source /home/$USER/miniconda3/etc/profile.d/conda.sh
#conda activate SR
######################################################

printf '%s\n' "Training Model on GPU ${CUDA_VISIBLE_DEVICES}"

python  train_video_superresolution.py \
  --model_type $model_type \
  --dataset reds_with_mv \
  --eval_datasets reds_with_mv \
  --num_blocks $num_blocks \
  --num_residual_units $num_residual_units \
  --scale $scale \
  --learning_rate 0.00020 \
  --train_batch_size $train_batch_size \
  --num_patches $num_patches \
  --lr_patch_size $lr_patch_size \
  --epochs $epochs \
  --image_batch 20 \
  --model_path '/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/runs/wdsr_b_x2_16_32_Dec22_21_04_48/block_index.txt' \
  --job_dir /data/zhuz/runs/$job_dir

