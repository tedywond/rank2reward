#!/bin/bash

#export NCCL_DEBUG=info
#export NCCL_P2P_DISABLE=1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


data_path=/net/scratch/teddy-shared/rank2reward
output_path=/home/tewodrosayalew/Video2Reward/output_dir/


srun torchrun \
--nproc_per_node=4 --master_port=${port} main.py --data_path ${data_path} --epochs 400 \
 --output_dir ${output_path} \
 --eval_path ${data_path} \
 --save_every 20 \
 --wandb_key e1da324bf2aa2b86e3e43934e9eff66eff117a71 \
 --batch_size 16
