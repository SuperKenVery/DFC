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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import model as Model
from discriminator import Discriminator
from data import InfiniteDIV2K, SRBenchmark

sys.path.insert(0, "../")  # run under the project directory
from common.option import GanFtOptions
from common.utils import PSNR, _rgb2ycbcr, logger_info
from common.Writer import Logger
from train_utils import round_func, SaveCheckpoint, valid_steps
from accelerate import Accelerator, DistributedDataParallelKwargs, logging
from accelerate.utils import ProjectConfiguration

if __name__ == "__main__":
    opt_inst = GanFtOptions()
    opt = opt_inst.parse()

    # Tensorboard for monitoring
    writer = Logger(log_dir=opt.logDir)

    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=opt.expDir,),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
        device_placement=False
    )
    logger_name = 'train'
    logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.log'))
    logger = logging.get_logger(logger_name)
    opt_inst.print_options(opt)

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, opt.model)

    generator_device, discriminator_device = "cuda:0", "cuda:1"
    model_G = model(sample_size=opt.sample_size, nf=opt.nf,
                    scale=opt.scale, modes=modes, stages=stages).to(generator_device)
    discriminator = Discriminator().to(discriminator_device)

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    params_D = list(filter(lambda p: p.requires_grad, discriminator.parameters()))
    opt_D = optim.Adam(params_D, lr=opt.lr0, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # LR
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
        opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize, deferred_load=True)
    train_loader = DataLoader(train_data, pin_memory=True,
                              num_workers=opt.numWorkers, batch_size=opt.batchSize)

    # Valid dataset
    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    model_G, opt_G, train_loader, scheduler = accelerator.prepare(
        model_G, opt_G, train_loader, scheduler)
    train_data.ensure_loaded()
    # Load saved params
    if opt.startIter > 0:
        load_path = f"{opt.expDir}/checkpoints/checkpoint_{opt.startIter}"
        if not os.path.exists(load_path):
            load_path = f"{opt.expDir}/checkpoints_gan/checkpoint_{opt.startIter}"
        accelerator.load_state(load_path)

    # Here things are different from 1_train_model.py.
    # Either model_G or the discriminator fits on a single A6000, but not both.
    # So we need to manually place them (so device_placement=False), and cannot use accelerate's auto placement.
    # But we still need accelerate.prepare for checkpoint loading with accelerator.load_state.
    discriminator, opt_D = accelerator.prepare(discriminator, opt_D)

    # GAN training functions
    def train_discriminator(real_hr, fake_hr):
        """Train discriminator to distinguish real from fake images
        Minimizes D(fake) - D(real), i.e., maximizes D(real) - D(fake)
        """
        opt_D.zero_grad()
        discriminator.train()
        model_G.eval()

        # Move to discriminator device
        real_hr_d = real_hr.detach().to(discriminator_device)
        fake_hr_d = fake_hr.detach().to(discriminator_device)

        # Real and fake predictions
        print(f"real_hr shape {real_hr_d.shape}, fake_hr shape {fake_hr_d.shape}")
        pred_real = discriminator(real_hr_d)
        pred_fake = discriminator(fake_hr_d)

        # Discriminator loss: minimize D(fake) - D(real)
        loss_D = (pred_fake - pred_real).mean()

        accelerator.backward(loss_D)
        opt_D.step()

        return loss_D.item(), pred_real.mean().item(), pred_fake.mean().item()

    def train_generator(im, lb):
        """Train generator with pixel loss and adversarial loss
        Minimizes -D(fake), i.e., maximizes D(fake)
        """
        opt_G.zero_grad()
        model_G.train()
        discriminator.eval()

        pred = model_G(im, 'train')

        # Pixel loss
        loss_pixel = F.mse_loss(pred, lb)

        # Adversarial loss: minimize -D(fake)
        pred_d = discriminator(pred.to(discriminator_device))
        loss_adv = -pred_d.mean()

        # Total generator loss
        loss_G = loss_pixel + opt.lambda_adv * loss_adv

        accelerator.backward(loss_G)
        opt_G.step()

        return pred, loss_G.item(), loss_pixel.item(), loss_adv.item()

    l_accum = [0., 0., 0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # TRAINING
    i = opt.startIter

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

    pbar = trange(opt.startIter + 1, opt.totalIter + 1)
    for i in pbar:
        model_G.train()

        # Data preparing
        st = time.time()
        im, lb = next(train_iter)
        im, lb = im.to(generator_device), lb.to(generator_device)
        dT += time.time() - st

        # TRAIN DISCRIMINATOR
        st = time.time()
        with torch.no_grad():
            fake_hr = model_G(im, 'train')

        loss_D, loss_D_real, loss_D_fake = train_discriminator(lb, fake_hr)

        # TRAIN GENERATOR
        pred, loss_G, loss_pixel, loss_adv = train_generator(im, lb)
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_G
        l_accum[1] += loss_pixel
        l_accum[2] += loss_adv
        l_accum[3] += loss_D
        l_accum[4] += loss_D_real

        # Show information
        if i % opt.displayStep == 0:
            writer.scalar_summary(
                'loss_G', l_accum[0] / opt.displayStep, i)
            writer.scalar_summary(
                'loss_Pixel', l_accum[1] / opt.displayStep, i)
            writer.scalar_summary(
                'loss_Adv', l_accum[2] / opt.displayStep, i)
            writer.scalar_summary(
                'loss_D', l_accum[3] / opt.displayStep, i)
            writer.scalar_summary(
                'loss_D_real', l_accum[4] / opt.displayStep, i)

            # Update progress bar with metrics
            metrics = {
                'G': f'{l_accum[0] / opt.displayStep:.2e}',
                'Pixel': f'{l_accum[1] / opt.displayStep:.2e}',
                'Adv': f'{l_accum[2] / opt.displayStep:.2e}',
                'D': f'{l_accum[3] / opt.displayStep:.2e}'
            }

            # Only show timing metrics if rT/dT <= 1000
            avg_dT = dT / opt.displayStep
            avg_rT = rT / opt.displayStep
            if avg_dT > 0 and avg_rT / avg_dT <= 1000:
                metrics['dT'] = f'{avg_dT:.4f}' # data loading time
                metrics['rT'] = f'{avg_rT:.4f}' # run time

            pbar.set_postfix(metrics)

            l_accum = [0., 0., 0., 0., 0.]
            dT = 0.
            rT = 0.

        # Save models
        if i % opt.saveStep == 0:
            accelerator.save_state(f"{opt.expDir}/checkpoints_gan/checkpoint_{i}")

        # Validation
        if i % opt.valStep == 0:
            valid_steps(model_G, valid, opt, i, writer, accelerator)

    logger.info("Complete")
