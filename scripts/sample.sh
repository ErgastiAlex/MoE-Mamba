cd projects/DiT/;

CUDA_VISIBLE_DEVICES=1

python3 sample.py --model DiT-B/2 --num-sampling-step 1000  \
         --ckpt /home/filippo/projects/DiT/results/000-DiT-B-2/checkpoints/0200000.pt