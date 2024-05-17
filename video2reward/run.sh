#!/bin/bash
#SBATCH --job-name=video2reward
#SBATCH --output=slurm/output.txt        
#SBATCH --error=slurm/error.txt        
#SBATCH --partition=dev-gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1      
#SBATCH --cpus-per-task=16          

hostname=$(hostname)
echo "Hostname: ${hostname}"

if [ "$hostname" = "maple" ]; then
    data_path=/home/kevinwu/ripl/Video2Reward/data/rank2reward
    base_output_path=/home/kevinwu/ripl/Video2Reward/output
else
    data_path=/share/data/ripl/rank2reward
    base_output_path=/home-nfs/kevinwu/ripl/Video2Reward/output
fi

subdir=$(date +%Y%m%d_%H%M%S)
# subdir="_norm_rand_aug"
output_path="${base_output_path}/${subdir}/"
mkdir -p ${output_path}
echo "Model checkpoint directory: ${output_path}"

torchrun main.py \
 --data_path ${data_path} \
 --epochs 500 \
 --output_dir ${output_path} \
 --eval_path ${data_path} \
 --save_every 50 \
 --wandb_key 486b8336215e9a9ef80ff65f033de41c99a2597f \
 --batch_size 16 \
 --augment \
#  --no_randomize \
#  --no_normalize_prediction \
