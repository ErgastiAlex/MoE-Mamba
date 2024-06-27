cd projects/DiT/;

CUDA_VISIBLE_DEVICES=1

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25678 train.py --model DiT-L/2 --data-path /home/filippo/datasets/afhq/train \
         --batch_size 1 --num-classes 3 #--use_mamba --c_input # --continue_train --ckpt /home/filippo/projects/DiT/results/000-DiT-B-2/checkpoints/0200000.pt \
        #  --train_steps 200000
