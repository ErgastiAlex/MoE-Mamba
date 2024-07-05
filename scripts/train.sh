cd projects/DiT/;

CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25633 train.py --model DiT-L/2 --data-path /home/filippo/datasets/afhq/train \
         --batch_size 1 --num-classes 3 --use_mamba --c_input --results_dir /home/filippo/projects/DiT/results/
 
          #--continue_train --train_steps 700000 --experiment_dir /home/filippo/projects/DiT/results/001-DiT-L-2--mamba \
         