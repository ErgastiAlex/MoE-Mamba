#!/bin/bash
#SBATCH --partition=gpu_vbd
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s_vbd:1
#SBATCH --qos=gpu_vbd
#SBATCH --time 8-00:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 4
#SBATCH --job-name=MM-DiT/Large-uncond/Moe
#SBATCH --output=%x.o%j

#< Charge resources to account
#SBATCH --account G_VBD

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate DiT

accelerate launch train.py --model DiT-B/1 --data-path /hpc/archive/G_VBD/alex.ergasti/datasets/celebaMM/ \
         --batch_size 8 --num-classes 0 \
         --results_dir /hpc/archive/G_VBD/alex.ergasti/MM-DiT/Large-uncond/No-MoE-log-8s \
         --use_mamba --learn_pos_emb \
         --sample-every 10000 \
         --ckpt-every 10000 \
         --use_ckpt \
         --sampling log
