from loguru import logger
from tqdm import tqdm, trange
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, logging
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs


import model as Model
from data import InfiniteDIV2K, SRBenchmark
from discriminator import Discriminator

sys.path.insert(0, "../")  # run under the project directory
from common.option import TrainOptions
from common.utils import PSNR, _rgb2ycbcr, logger_info
from common.Writer import Logger
from train_utils import round_func, SaveCheckpoint, valid_steps



def train_discriminator(model_G, model_D, opt_D, real_imgs, fake_imgs, accelerator):
    """
    Train discriminator to distinguish real from fake images

    Args:
        model_G: Generator model
        model_D: Discriminator model
        opt_D: Discriminator optimizer
        real_imgs: Real high-resolution images
        fake_imgs: Generated fake images (detached)
        accelerator: Accelerator instance

    Returns:
        loss_D: Discriminator loss value
    """
    # Set generator to eval mode, discriminator to train mode
    model_G.eval()
    model_D.train()

    opt_D.zero_grad()

    # Get discriminator loss for real images (target=1)
    loss_D_real = model_D(real_imgs, for_real=True)

    # Get discriminator loss for fake images (target=0)
    loss_D_fake = model_D(fake_imgs.detach(), for_real=False)

    loss_D = (loss_D_real + loss_D_fake).mean()

    accelerator.backward(loss_D)
    opt_D.step()

    return loss_D.item()


def train_generator(model_G, model_D, opt_G, im, lb, accelerator, gan_weight=0.01):
    """
    Train generator with both reconstruction loss and adversarial loss

    Args:
        model_G: Generator model
        model_D: Discriminator model
        opt_G: Generator optimizer
        im: Low-resolution input images
        lb: High-resolution target images
        accelerator: Accelerator instance
        gan_weight: Weight for adversarial loss

    Returns:
        loss_G_total: Total generator loss
        loss_pixel: Pixel-wise reconstruction loss
        loss_G_adv: Adversarial loss
    """
    # Set generator to train mode, discriminator to eval mode for stable gradients
    model_G.train()
    model_D.eval()

    opt_G.zero_grad()

    # Generate super-resolved images
    pred = model_G(im, 'train')

    # Pixel-wise reconstruction loss
    loss_pixel = F.mse_loss(pred, lb)

    # Adversarial loss - generator tries to fool discriminator
    loss_G_adv = model_D(pred, for_G=True).mean()

    # Total generator loss
    loss_G_total = loss_pixel + gan_weight * loss_G_adv

    accelerator.backward(loss_G_total)
    opt_G.step()

    return loss_G_total.item(), loss_pixel.item(), loss_G_adv.item()


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    # Tensorboard for monitoring
    writer = Logger(log_dir=opt.logDir)

    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=opt.expDir,),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )

    logger_name = 'train'
    logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.log'))
    logger = logging.get_logger(logger_name)
    opt_inst.print_options(opt)

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, opt.model)

    model_G = model(sample_size=opt.sample_size, nf=opt.nf,
                    scale=opt.scale, modes=modes, stages=stages)

    # Optimizers for generator
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    # LR scheduler for generator
    if opt.lr1 < 0:
        def lf(x): return (
            ((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b

        def lf(x): return (
            ((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Training dataset
    train_data = InfiniteDIV2K(
        opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize)
    train_loader = DataLoader(train_data, pin_memory=True,
                              num_workers=opt.workerNum, batch_size=opt.batchSize)

    # Valid dataset
    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    # Prepare generator, optimizer, dataloader first (BEFORE loading state)
    model_G, opt_G, train_loader, scheduler = accelerator.prepare(
        model_G, opt_G, train_loader, scheduler)

    # Load saved params for generator (if exists)
    # Try loading from checkpoints_gan first, then fall back to checkpoints
    if opt.startIter > 0:
        gan_checkpoint_path = f"{opt.expDir}/checkpoints_gan/checkpoint_{opt.startIter}"
        regular_checkpoint_path = f"{opt.expDir}/checkpoints/checkpoint_{opt.startIter}"

        if os.path.exists(gan_checkpoint_path):
            logger.info(f"Loading GAN checkpoint from {gan_checkpoint_path}")
            accelerator.load_state(gan_checkpoint_path)
        elif os.path.exists(regular_checkpoint_path):
            logger.info(f"Loading regular checkpoint from {regular_checkpoint_path}")
            accelerator.load_state(regular_checkpoint_path)
        else:
            logger.warning(f"No checkpoint found at iteration {opt.startIter}")

    # NOW create and prepare discriminator (AFTER loading state)
    model_D = Discriminator(device=accelerator.device, precision="bf16")

    # Optimizer for discriminator
    params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
    opt_D = optim.Adam(params_D, lr=opt.lr0 * 0.1, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # Prepare discriminator
    model_D, opt_D = accelerator.prepare(model_D, opt_D)

    l_accum = [0., 0., 0., 0.]  # [pixel_loss, adv_loss, D_loss, total_G_loss]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # TRAINING
    i = opt.startIter

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

    pbar = trange(opt.startIter + 1, opt.totalIter + 1)
    for i in pbar:
        # Data preparing
        st = time.time()
        im, lb = next(train_iter)
        dT += time.time() - st

        # TRAIN
        st = time.time()

        # Train Generator
        loss_G_total, loss_pixel, loss_G_adv = train_generator(
            model_G, model_D, opt_G, im, lb, accelerator, gan_weight=1)

        # Train Discriminator
        with torch.no_grad():
            fake_imgs = model_G(im, 'train')
        loss_D = train_discriminator(model_G, model_D, opt_D, lb, fake_imgs, accelerator)

        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_pixel
        l_accum[1] += loss_G_adv
        l_accum[2] += loss_D
        l_accum[3] += loss_G_total

        # Show information
        if i % opt.displayStep == 0:
            writer.scalar_summary('loss_Pixel', l_accum[0] / opt.displayStep, i)
            writer.scalar_summary('loss_G_Adv', l_accum[1] / opt.displayStep, i)
            writer.scalar_summary('loss_D', l_accum[2] / opt.displayStep, i)
            writer.scalar_summary('loss_G_Total', l_accum[3] / opt.displayStep, i)

            # Prepare metrics for progress bar
            metrics = {
                'GPixel': f'{l_accum[0] / opt.displayStep:.2e}',
                'GAdv': f'{l_accum[1] / opt.displayStep:.2e}',
                'D': f'{l_accum[2] / opt.displayStep:.2e}',
                'GTotal': f'{l_accum[3] / opt.displayStep:.2e}',
            }

            # Conditionally add timing metrics
            avg_dT = dT / opt.displayStep
            avg_rT = rT / opt.displayStep
            if avg_dT > 0 and avg_rT / avg_dT < 1000:
                metrics['dT'] = f'{avg_dT:.4f}'
                metrics['rT'] = f'{avg_rT:.4f}'

            pbar.set_postfix(metrics)

            l_accum = [0., 0., 0., 0.]
            dT = 0.
            rT = 0.

        # Save models to checkpoints_gan
        if i % opt.saveStep == 0:
            accelerator.save_state(f"{opt.expDir}/checkpoints_gan/checkpoint_{i}")

        # Validation
        if i % opt.valStep == 0:
            valid_steps(model_G, valid, opt, i, writer, accelerator)

    logger.info("Complete")
