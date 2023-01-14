#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Experiments

model_type='BASIC_MODEL'  # NAS_MODEL / BASIC_MODEL

epochs=30
num_patches=200    # default 1000
train_batch_size=16 # default 16
lr_patch_size=96    # default 48

# arch
scale=2
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

python  pretrain_simplified_model.py \
  --model_type $model_type \
  --dataset div2k \
  --eval_datasets urban100 \
  --num_blocks $num_blocks \
  --num_residual_units $num_residual_units \
  --scale $scale \
  --train_batch_size $train_batch_size \
  --num_patches $num_patches \
  --lr_patch_size $lr_patch_size \
  --epochs $epochs \
  --model_path '/home/zhuzhui/super-resolution/MyNAS/compiler-aware-nas-sr/runs/wdsr_b_x2_16_32_Nov27_19_01_41/block_index.txt' \
  --job_dir runs/$job_dir

