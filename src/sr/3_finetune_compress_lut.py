import datetime
import math
import time

import safetensors
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
from .train_utils import get_lut_config, valid_steps

torch.backends.cudnn.benchmark = True


def main(accelerator: Accelerator, exp: Experiment, writer, logger):
    config = exp.config
    ft_config = config.finetune_lut

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
        lr=ft_config.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.train.weight_decay,
        amsgrad=False,
    )

    # Learning rate schedule
    total_iter = ft_config.total_iter
    if ft_config.lr1 < 0:

        def lf(x):
            return (((1 + math.cos(x * math.pi / total_iter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = ft_config.lr1 / ft_config.lr0
        lr_a = 1 - lr_b

        def lf(x):
            return (((1 + math.cos(x * math.pi / total_iter)) / 2) ** 1.0) * lr_a + lr_b

    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Training dataset
    with accelerator.main_process_first():
        train_data = InfiniteDIV2K(
            ft_config.batch_size,
            config.data.worker_num,
            config.model.scale,
            config.data.train_dir,
            config.data.crop_size,
        )
    train_loader = DataLoader(
        train_data,
        pin_memory=True,
        num_workers=config.data.worker_num,
        batch_size=ft_config.batch_size,
    )

    # Valid dataset
    valid = SRBenchmark(config.data.val_dir, scale=config.model.scale)

    model_G, opt_G, train_loader, scheduler = accelerator.prepare(
        model_G, opt_G, train_loader, scheduler
    )

    lut_cfg = get_lut_config(config.export_lut, config.model.interval)
    umodel = accelerator.unwrap_model(model_G)

    start_iter = ft_config.start_iter
    if start_iter == 0:
        # Load exported lut
        lut_path = (
            exp.get_lut_checkpoint_path(ft_config.export_lut_iter) / "model.safetensors"
        )
        state_dict = safetensors.torch.load_file(str(lut_path))
    else:
        # Load finetune checkpoint
        lutft_path = exp.get_lutft_checkpoint_path(start_iter) / "model.safetensors"
        state_dict = safetensors.torch.load_file(str(lutft_path))

    with umodel.load_state_from_lut(lut_cfg, accelerator):
        umodel.load_state_dict(state_dict)

    if start_iter == 0:
        valid_steps(model_G, valid, exp, 0, writer, accelerator)

    l_accum = [0.0, 0.0, 0.0]
    dT = 0.0
    rT = 0.0
    accum_samples = 0

    # Create iterator for infinite dataset
    train_iter = iter(train_loader)

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
        accum_samples += ft_config.batch_size
        l_accum[0] += loss_G.item()

        # Show information
        if i % ft_config.display_step == 0:
            writer.scalar_summary("loss_Pixel", l_accum[0] / ft_config.display_step, i)

            logger.info(
                "{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                    exp.exp_dir,
                    i,
                    accum_samples,
                    l_accum[0] / ft_config.display_step,
                    dT / ft_config.display_step,
                    rT / ft_config.display_step,
                )
            )
            l_accum = [0.0, 0.0, 0.0]
            dT = 0.0
            rT = 0.0

        # Save models
        if i % ft_config.save_step == 0:
            save_path = exp.get_lutft_checkpoint_path(i)
            with umodel.save_as_lut(lut_cfg):
                accelerator.save_model(model_G, str(save_path))

        # Validation
        if i % ft_config.val_step == 0:
            valid_steps(model_G, valid, exp, i, writer, accelerator)

    logger.info("Complete")


if __name__ == "__main__":
    exp = load_experiment()
    exp.ensure_dirs()

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
                / f"lut_finetune_{datetime.datetime.now()}_rank{accelerator.process_index}.log"
            ),
        )
        logger = logging.get_logger(logger_name)
        exp.print_config(logger)

    # Tensorboard for monitoring
    writer = Logger(log_dir=str(exp.log_dir))

    try:
        main(accelerator, exp, writer, logger)
    except BaseException:
        if accelerator.is_main_process:
            raise
