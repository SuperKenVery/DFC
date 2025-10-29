from train_utils import round_func, SaveCheckpoint, valid_steps
from common.Writer import Logger
from common.utils import PSNR, _rgb2ycbcr
from common.option import TrainOptions
from tqdm import tqdm, trange
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from accelerate import logging

import model as Model
from data import InfiniteDIV2K, SRBenchmark, rigid_aug
from common.utils import logger_info

sys.path.insert(0, "../")  # run under the project directory

torch.backends.cudnn.benchmark = True

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    # Tensorboard for monitoring
    writer = Logger(log_dir=opt.logDir)

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=opt.expDir,
        ),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )

    with accelerator.main_process_first():
        logger_name = "train"
        logger_info(logger_name, os.path.join(opt.expDir, logger_name + ".log"))
        logger = logging.get_logger(logger_name)
        opt_inst.print_options(opt)

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, opt.model)

    model_G = model(
        sample_size=opt.sample_size,
        nf=opt.nf,
        scale=opt.scale,
        modes=modes,
        stages=stages,
    )

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(
        params_G,
        lr=opt.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt.weightDecay,
        amsgrad=False,
    )

    # LR
    if opt.lr1 < 0:

        def lf(x):
            return (
                ((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0
            ) * 0.8 + 0.2
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b

        def lf(x):
            return (
                ((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0
            ) * lr_a + lr_b

    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Load saved params
    if opt.startIter > 0:
        accelerator.load_state(f"{opt.expDir}/checkpoints/checkpoint_{opt.startIter}")

    # Training dataset
    with accelerator.main_process_first():
        train_data = InfiniteDIV2K(
            opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize
        )
    train_loader = DataLoader(
        train_data,
        pin_memory=True,
        num_workers=opt.workerNum,
        batch_size=opt.batchSize,
    )

    # Valid dataset
    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    model_G, opt_G, train_loader, scheduler = accelerator.prepare(
        model_G, opt_G, train_loader, scheduler
    )

    l_accum = [0.0, 0.0, 0.0]
    dT = 0.0
    rT = 0.0
    accum_samples = 0

    # TRAINING
    i = opt.startIter

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

    for i in trange(opt.startIter + 1, opt.totalIter + 1, dynamic_ncols=True):
        model_G.train()

        # Data preparing
        st = time.time()
        batch = next(train_iter)
        im, lb = rigid_aug(batch)
        dT += time.time() - st

        # TRAIN G
        st = time.time()
        opt_G.zero_grad()

        pred = model_G(im, "train")

        loss_G = F.mse_loss(pred, lb)
        accelerator.backward(loss_G)
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()

        # Show information
        if i % opt.displayStep == 0:
            writer.scalar_summary("loss_Pixel", l_accum[0] / opt.displayStep, i)

            logger.info(
                "{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                    opt.expDir,
                    i,
                    accum_samples,
                    l_accum[0] / opt.displayStep,
                    dT / opt.displayStep,
                    rT / opt.displayStep,
                )
            )
            l_accum = [0.0, 0.0, 0.0]
            dT = 0.0
            rT = 0.0

        # Save models
        if i % opt.saveStep == 0:
            accelerator.save_state(f"{opt.expDir}/checkpoints/checkpoint_{i}")

        # Validation
        if i % opt.valStep == 0:
            valid_steps(model_G, valid, opt, i, writer, accelerator)

    logger.info("Complete")
