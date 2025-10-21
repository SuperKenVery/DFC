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
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, TorchDynamoPlugin, DynamoBackend


import model as Model
from data import InfiniteDIV2K, SRBenchmark
from discriminator import Discriminator

sys.path.insert(0, "../")  # run under the project directory
from common.option import GanFtOptions, TrainOptions
from common.utils import PSNR, _rgb2ycbcr, logger_info, multiline_tqdm
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
        num_correct: Number of correct predictions (D(real) > D(fake))
        sum_advantage: Sum of advantages (D(real) - D(fake))
    """

    opt_D.zero_grad()

    loss_D_real, score_real = model_D(real_imgs, for_real=True)
    loss_D_fake, score_fake = model_D(fake_imgs.detach(), for_real=False)
    loss_D = (loss_D_real + loss_D_fake).mean()

    accelerator.backward(loss_D)
    opt_D.step()

    # Compute metrics using raw scores
    with torch.no_grad():
        # Higher score = more "real"
        # Discriminator is correct when score_real > score_fake
        correct = (score_real > score_fake).float()
        num_correct = correct.sum().item()

        # Advantage: how much higher the score is for real vs fake
        # Positive advantage means D is performing well
        advantage = score_real - score_fake
        sum_advantage = advantage.sum().item()

    return num_correct, sum_advantage


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

    opt_G.zero_grad()

    pred = model_G(im, 'train')

    loss_pixel = F.mse_loss(pred, lb)
    loss_G_adv, _ = model_D(pred, for_G=True)
    loss_G_total = loss_pixel + gan_weight * loss_G_adv.mean()

    accelerator.backward(loss_G_total)
    opt_G.step()

    return loss_G_total.item(), loss_pixel.item(), loss_G_adv.mean().item()


if __name__ == "__main__":
    opt_inst = GanFtOptions()
    opt = opt_inst.parse()

    # Tensorboard for monitoring
    writer = Logger(log_dir=opt.logDir)

    torch.backends.cuda.matmul.allow_tf32 = True
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
    model_D = Discriminator(device=accelerator.device)

    # Optimizer for discriminator
    params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
    opt_D = optim.Adam(params_D, lr=opt.lr0 * 0.1, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # Prepare discriminator
    model_D, opt_D = accelerator.prepare(model_D, opt_D)

    l_accum = [0., 0., 0., 0., 0.]  # [pixel_loss, adv_loss, D_accuracy, D_advantage, total_G_loss]
    dT = 0.
    rT = 0.

    # TRAINING
    i = opt.startIter

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

    pbar = multiline_tqdm(range(opt.startIter + 1, opt.totalIter + 1), dynamic_ncols=True)
    pbar_postfix = {}
    for i in pbar:
        # Data preparing
        st = time.time()
        im, lb = next(train_iter)
        dT += time.time() - st

        # TRAIN
        model_G.train()
        model_D.train()
        st = time.time()

        # Train Generator
        if i > opt.startIter + opt.dInitSteps:
            loss_G_total, loss_pixel, loss_G_adv = train_generator(
                model_G, model_D, opt_G, im, lb, accelerator, gan_weight=opt.ganWeight)

            l_accum[0] += loss_pixel
            l_accum[1] += loss_G_adv
            l_accum[4] += loss_G_total

        # Train Discriminator
        with torch.no_grad():
            fake_imgs = model_G(im, 'train')
        num_correct, sum_advantage = train_discriminator(model_G, model_D, opt_D, lb, fake_imgs, accelerator)

        scheduler.step()

        rT += time.time() - st

        # For monitoring
        l_accum[2] += num_correct
        l_accum[3] += sum_advantage

        # Log input images and discriminator scores to tensorboard
        if i % opt.viewDStep == 0:
            model_G.eval()
            model_D.eval()
            with torch.no_grad():
                # Generate fake images
                fake_for_logging = model_G(im, 'train')

                # Get discriminator scores (use raw scores, not losses)
                _, d_score_real = model_D(lb, for_real=True)
                _, d_score_fake = model_D(fake_for_logging, for_real=False)
                d_score_real = d_score_real.mean()
                d_score_fake = d_score_fake.mean()

                # Log images (clamp to [0, 1] range and take first 4 samples from batch)
                num_log_imgs = min(4, im.size(0))
                writer.image_summary('images/D_LR_input', im[:num_log_imgs].clamp(0, 1), i)
                writer.image_summary('images/D_HR_target', lb[:num_log_imgs].clamp(0, 1), i)
                writer.image_summary('images/D_SR_generated', fake_for_logging[:num_log_imgs].clamp(0, 1), i)

                # Log discriminator scores
                writer.scalar_summary('discriminator/score_real', d_score_real.item(), i)
                writer.scalar_summary('discriminator/score_fake', d_score_fake.item(), i)

            model_G.train()
            model_D.train()

        # Show information
        if i % opt.displayStep == 0:
            # Calculate discriminator metrics
            total_samples = opt.displayStep * opt.batchSize
            d_accuracy = (l_accum[2] / total_samples) * 100  # Convert to percentage
            d_advantage = l_accum[3] / total_samples

            writer.scalar_summary('loss_Pixel', l_accum[0] / opt.displayStep, i)
            writer.scalar_summary('loss_G_Adv', l_accum[1] / opt.displayStep, i)
            writer.scalar_summary('discriminator/accuracy', d_accuracy, i)
            writer.scalar_summary('discriminator/advantage', d_advantage, i)
            writer.scalar_summary('loss_G_Total', l_accum[4] / opt.displayStep, i)

            # Prepare metrics for progress bar
            pbar_postfix.update({
                'GPixel': f'{l_accum[0] / opt.displayStep:.3e}',
                'GAdv': f'{l_accum[1] / opt.displayStep:.3e}',
                'DAcc': f'{d_accuracy:.1f}%',
                'DAdv': f'{d_advantage:.3f}',
                'GTotal': f'{l_accum[4] / opt.displayStep:.3e}',
            })

            # Conditionally add timing metrics
            avg_dT = dT / opt.displayStep
            avg_rT = rT / opt.displayStep
            if avg_dT > 0 and avg_rT / avg_dT < 1000:
                pbar_postfix.update({
                    'dT': f'{avg_dT:.4f}',
                    'rT': f'{avg_rT:.4f}'
                })

            pbar.set_postfix(pbar_postfix)

            l_accum = [0., 0., 0., 0., 0.]
            dT = 0.
            rT = 0.

        # Save models to checkpoints_gan
        if i % opt.saveStep == 0:
            logger.info(f"Saved model at step {i}")
            accelerator.save_state(f"{opt.expDir}/checkpoints_gan/checkpoint_{i}")

        # Validation
        if i > opt.startIter + opt.dInitSteps and i % opt.valStep == 0:
            psnrs = valid_steps(model_G, valid, opt, i, writer, accelerator)
            pbar_postfix.update(psnrs)
            pbar.set_postfix(psnrs)

    logger.info("Complete")
