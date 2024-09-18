# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from torchvision.utils import save_image
from glob import glob
from time import time
import argparse
import logging
import os
from download import find_model

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusion.rectified_flow import RectifiedFlow


from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    try:
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [int(x.split("_")[-1]) for x in checkpoint_files]
        checkpoint_files.sort()
        checkpoint_file = checkpoint_files[-1]
    except:
        checkpoint_file = 0

    accelerator_project_config = ProjectConfiguration(project_dir=experiment_dir, automatic_checkpoint_naming=True, iteration = checkpoint_file+1)
    accelerator = Accelerator(project_config=accelerator_project_config)
    
    set_seed(args.global_seed)  # Set global seed for reproducibility


    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=False,
        use_checkpoint=True,
        use_mamba=args.use_mamba,
        use_moe=args.use_moe,
        learn_pos_emb=args.learn_pos_emb
    )

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model


    if args.use_rf:
        rectified_flow = RectifiedFlow(num_timesteps=25)
    else:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    vae, model, ema, opt, loader = accelerator.prepare(vae, model, ema, opt, loader)


  # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    try:
        accelerator.load_state()
        logger.info(f"Checkpoint found, starting from {checkpoint_file*args.ckpt_every} steps.")
        train_steps = checkpoint_file*args.ckpt_every
        log_steps = train_steps % args.log_every
    except:
        # accelerator.save_state()
        logger.info("No checkpoint found, starting from scratch.")
    
    device = accelerator.device 

    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)
            if args.use_rf:
                loss_dict = rectified_flow.training_loss(model, x, model_kwargs)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)

            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                accelerator.save_state()
                logger.info(f"Checkpoint saved at step {train_steps}.")
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                with torch.no_grad():
                    model.eval()
                    device = x.device

                    class_labels = [0,1,2,0,1,2,0,1]

                    # Create sampling noise:
                    n = len(class_labels)
                    if args.use_rf:
                        z = torch.randn(n, 4, latent_size, latent_size, device=device)*rectified_flow.noise_scale
                    else:
                        z = torch.randn(n, 4, latent_size, latent_size, device=device)

                    y = torch.tensor(class_labels, device=device)

                    # Setup classifier-free guidance:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([args.num_classes] * n, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=1.0)


                    # Sample images:
                    if args.use_rf:
                        samples = rectified_flow.sample(
                            model.forward_with_cfg, z, model_kwargs=model_kwargs, progress=True)
                    else:
                        samples = diffusion.p_sample_loop(
                            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)

                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                    samples = vae.decode(samples / 0.18215).sample

                    # Save and display images:
                    save_image(samples, f"{experiment_dir}/sample_{train_steps}.png", nrow=4, normalize=True, value_range=(-1, 1))
                    model.train()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    checkpoint_path = f"{checkpoint_dir}/last.pt"
    accelerator.save({
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": accelerator.unwrap_model(ema).state_dict(),
            "opt": opt.optimizer.state_dict(),
            "args": args
        }, checkpoint_path)
    logger.info(f"Saved last to {checkpoint_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--experiment_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--learn-pos-emb", action='store_true')

    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument('--use_mamba', action='store_true') 
    parser.add_argument('--use_moe', action='store_true')

    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--use_rf", action='store_true')
    
    args = parser.parse_args()
    main(args)
