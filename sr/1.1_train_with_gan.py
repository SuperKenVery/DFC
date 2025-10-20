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

    # CUDA streams for pipelined execution
    stream_gen = torch.cuda.Stream(device=torch.device(generator_device))  # Background generation
    stream_disc = torch.cuda.Stream(device=torch.device(discriminator_device))  # Discriminator training
    # Generator training uses default stream on GPU0

    # CUDA events for synchronization between streams
    event_gen_complete = [torch.cuda.Event() for _ in range(2)]  # Double buffer events

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

    l_accum = [0., 0., 0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # TRAINING
    i = opt.startIter

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

    # Double buffering: pre-generate first 2 batches
    buffers = [
        {'im': None, 'lb': None, 'fake_hr': None},
        {'im': None, 'lb': None, 'fake_hr': None}
    ]

    # Pre-fill buffers
    for buf_idx in range(2):
        im, lb = next(train_iter)
        im = im.to(generator_device, non_blocking=True)
        lb = lb.to(generator_device, non_blocking=True)

        with torch.cuda.stream(stream_gen):
            with torch.no_grad():
                fake_hr = model_G(im, 'train')
            event_gen_complete[buf_idx].record(stream_gen)

        buffers[buf_idx] = {'im': im, 'lb': lb, 'fake_hr': fake_hr}

    pbar = trange(opt.startIter + 1, opt.totalIter + 1, dynamic_ncols=True)
    for i in pbar:
        current_idx = i % 2
        next_idx = (i + 1) % 2

        # Data preparing timing
        st = time.time()

        # Wait for current buffer's generation to complete
        event_gen_complete[current_idx].synchronize()

        # Get current batch from buffer
        im = buffers[current_idx]['im']
        lb = buffers[current_idx]['lb']
        fake_hr = buffers[current_idx]['fake_hr']

        dT += time.time() - st

        st = time.time()

        # === GENERATOR FORWARD PASS (with gradients) ===
        # This produces pred for pixel loss and adversarial loss
        opt_G.zero_grad()
        model_G.train()
        discriminator.eval()  # Set discriminator to eval before generator forward
        pred = model_G(im, 'train')
        loss_pixel = F.mse_loss(pred, lb)

        # === DISCRIMINATOR TRAINING on GPU1 (parallel) ===
        with torch.cuda.stream(stream_disc):
            discriminator.train()
            opt_D.zero_grad()

            # Move to discriminator device
            real_hr_d = lb.detach().to(discriminator_device, non_blocking=True)
            fake_hr_d = fake_hr.detach().to(discriminator_device, non_blocking=True)

            # Real and fake predictions
            pred_real = discriminator(real_hr_d)
            pred_fake = discriminator(fake_hr_d)

            # Discriminator loss
            loss_D = (pred_fake - pred_real).mean()
            accelerator.backward(loss_D)
            opt_D.step()

            loss_D_val = loss_D.item()
            loss_D_real_val = pred_real.mean().item()
            loss_D_fake_val = pred_fake.mean().item()

        # === GENERATE NEXT BATCH on GPU0 (parallel with discriminator) ===
        if i < opt.totalIter:
            with torch.cuda.stream(stream_gen):
                im_next, lb_next = next(train_iter)
                im_next = im_next.to(generator_device, non_blocking=True)
                lb_next = lb_next.to(generator_device, non_blocking=True)

                # Generate fake images for next discriminator training (no gradients needed)
                with torch.no_grad():
                    fake_hr_next = model_G(im_next, 'train')

                event_gen_complete[next_idx].record(stream_gen)
                buffers[next_idx] = {'im': im_next, 'lb': lb_next, 'fake_hr': fake_hr_next}

        # === DISCRIMINATOR FORWARD for adversarial loss (on GPU1) ===
        # Wait for discriminator training to finish before using it for adversarial loss
        stream_disc.synchronize()

        # Discriminator already in eval mode, gradients flow to generator but not to discriminator params
        pred_d = discriminator(pred.to(discriminator_device, non_blocking=True))
        loss_adv = -pred_d.mean()

        # === GENERATOR BACKWARD ===
        loss_G = loss_pixel.to(discriminator_device) + opt.lambda_adv * loss_adv
        accelerator.backward(loss_G)
        opt_G.step()
        scheduler.step()

        loss_G_val = loss_G.item()
        loss_pixel_val = loss_pixel.item()
        loss_adv_val = loss_adv.item()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_G_val
        l_accum[1] += loss_pixel_val
        l_accum[2] += loss_adv_val
        l_accum[3] += loss_D_val
        l_accum[4] += loss_D_real_val

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
            # Synchronize all streams before saving
            stream_gen.synchronize()
            stream_disc.synchronize()
            torch.cuda.current_stream(device=generator_device).synchronize()
            accelerator.save_state(f"{opt.expDir}/checkpoints_gan/checkpoint_{i}")

        # Validation
        if i % opt.valStep == 0:
            # Synchronize all streams before validation
            stream_gen.synchronize()
            stream_disc.synchronize()
            torch.cuda.current_stream(device=generator_device).synchronize()
            valid_steps(model_G, valid, opt, i, writer, accelerator)

    logger.info("Complete")
