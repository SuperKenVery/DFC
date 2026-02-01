import datetime
import math
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs, logging
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import trange

from ..common.config import Experiment, load_experiment
from ..common.utils import logger_info
from ..common.Writer import Logger
from . import model as Model
from .data import InfiniteDIV2K, SRBenchmark, rigid_aug
from .train_utils import valid_steps

torch.backends.cudnn.benchmark = True


def main(accelerator: Accelerator, exp: Experiment, logger):
    config = exp.config

    model_cls = getattr(Model, config.model.model)

    model_G = model_cls(
        sample_size=config.model.sample_size,
        nf=config.model.nf,
        scale=config.model.scale,
        branches=config.model.branches,
        stages=config.model.stages,
    )

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(
        params_G,
        lr=config.train.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.train.weight_decay,
        amsgrad=False,
    )

    # Learning rate schedule
    total_iter = config.train.total_iter
    if config.train.lr1 < 0:

        def lf(x):
            return (((1 + math.cos(x * math.pi / total_iter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = config.train.lr1 / config.train.lr0
        lr_a = 1 - lr_b

        def lf(x):
            return (((1 + math.cos(x * math.pi / total_iter)) / 2) ** 1.0) * lr_a + lr_b

    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Training dataset
    with accelerator.main_process_first():
        train_data = InfiniteDIV2K(
            config.data.batch_size,
            config.data.worker_num,
            config.model.scale,
            config.data.train_dir,
            config.data.crop_size,
        )
    train_loader = DataLoader(
        train_data,
        pin_memory=True,
        num_workers=config.data.worker_num,
        batch_size=config.data.batch_size,
    )

    # Valid dataset
    valid = SRBenchmark(config.data.val_dir, scale=config.model.scale)

    model_G, opt_G, train_loader, scheduler = accelerator.prepare(
        model_G, opt_G, train_loader, scheduler
    )

    # Load saved params
    start_iter = config.train.start_iter
    if start_iter > 0:
        accelerator.load_state(str(exp.get_checkpoint_path(start_iter)))

    l_accum = [0.0, 0.0, 0.0]
    dT = 0.0
    rT = 0.0
    accum_samples = 0

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

    # Tensorboard for monitoring
    writer = Logger(log_dir=str(exp.log_dir))

    for i in trange(start_iter + 1, total_iter + 1, dynamic_ncols=True):
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
        accum_samples += config.data.batch_size
        l_accum[0] += loss_G.item()

        # Show information
        if i % config.train.display_step == 0:
            writer.scalar_summary(
                "loss_Pixel", l_accum[0] / config.train.display_step, i
            )

            logger.info(
                "{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                    exp.exp_dir,
                    i,
                    accum_samples,
                    l_accum[0] / config.train.display_step,
                    dT / config.train.display_step,
                    rT / config.train.display_step,
                )
            )
            l_accum = [0.0, 0.0, 0.0]
            dT = 0.0
            rT = 0.0

        # Save models
        if i % config.train.save_step == 0:
            accelerator.save_state(str(exp.get_checkpoint_path(i)))

        # Validation
        if i % config.train.val_step == 0:
            valid_steps(model_G, valid, exp, i, writer, accelerator)

    logger.info("Complete")


if __name__ == "__main__":
    exp = load_experiment()
    exp.ensure_dirs()
    exp.save_code()

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=str(exp.exp_dir),
        ),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )

    with accelerator.main_process_first():
        logger_name = "train"
        logger_info(
            logger_name,
            str(
                exp.exp_dir
                / f"{logger_name}_{datetime.datetime.now()}_rank{accelerator.process_index}.log"
            ),
        )
        logger = logging.get_logger(logger_name)
        exp.print_config(logger)

    try:
        main(accelerator, exp, logger)
    except BaseException:
        if accelerator.is_main_process:
            raise
