from train_utils import round_func, SaveCheckpoint, valid_steps
from common.Writer import Logger
from common.utils import PSNR, _rgb2ycbcr
from common.option import TrainOptions
from common.lut_module import LUTConfig, DFCConfig
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
import datetime
import model as Model
from data import InfiniteDIV2K, SRBenchmark, rigid_aug
from common.utils import logger_info
from train_utils import get_lut_cfg

sys.path.insert(0, "../")  # run under the project directory

torch.backends.cudnn.benchmark = True

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}


def main(accelerator: Accelerator, opt, logger):
    modes: list[str] = [i for i in opt.modes]
    stages: int = opt.stages

    model: type = getattr(Model, opt.model)

    model_G: torch.nn.Module = model(
        sample_size=opt.sample_size,
        nf=opt.nf,
        scale=opt.scale,
        modes=modes,
        stages=stages,
    )

    model_G = accelerator.prepare(model_G)

    # Load saved params
    assert opt.startIter > 0, "Please specify a iter to load"
    ckpt_dir = f"{opt.expDir}/checkpoints/checkpoint_{opt.startIter}"
    accelerator.load_state(ckpt_dir)

    lut_cfg = get_lut_cfg(opt)
    with accelerator.unwrap_model(model_G).save_as_lut(lut_cfg):
        lut_ckpt_dir = f"{ckpt_dir}/lut"
        accelerator.save_model(model_G, lut_ckpt_dir)

    logger.info("Complete")


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
        logger_info(
            logger_name,
            os.path.join(
                opt.expDir,
                f"{logger_name} {datetime.datetime.now()} rank={accelerator.process_index}.log",
            ),
        )
        logger = logging.get_logger(logger_name)
        opt_inst.print_options(opt)

    try:
        main(accelerator, opt, logger)
    except BaseException:
        if accelerator.is_main_process:
            raise
