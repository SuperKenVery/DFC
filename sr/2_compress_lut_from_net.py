import datetime
import os
import sys
import warnings
from pathlib import Path

import model as Model
import safetensors
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, logging
from accelerate.utils import ProjectConfiguration
from train_utils import get_lut_cfg, valid_steps

from common.option import TrainOptions
from common.utils import logger_info
from common.Writer import Logger
from data import SRBenchmark  # pyright: ignore[reportAttributeAccessIssue]

sys.path.insert(0, "../")  # run under the project directory
torch.backends.cudnn.benchmark = True
mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}
warnings.simplefilter(action="ignore", category=FutureWarning)


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
    umodel = accelerator.unwrap_model(model_G)

    # Load saved params
    assert opt.startIter > 0, "Please specify a iter to load"
    ckpt_dir = f"{opt.expDir}/checkpoints/checkpoint_{opt.startIter}"
    accelerator.load_state(ckpt_dir)

    lut_cfg = get_lut_cfg(opt)
    with umodel.save_as_lut(lut_cfg):
        lut_ckpt_dir = f"{ckpt_dir}/lut"
        accelerator.save_model(model_G, lut_ckpt_dir)

    # Test exported model
    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    logger.info("Original model before exporting:")
    valid_steps(model_G, valid, opt, 0, writer, accelerator)

    state_dict = safetensors.torch.load_file(f"{ckpt_dir}/lut/model.safetensors")  # pyright: ignore[reportAttributeAccessIssue]
    with umodel.load_state_from_lut(lut_cfg, accelerator):
        umodel.load_state_dict(state_dict)

    logger.info("Exported model:")
    valid_steps(model_G, valid, opt, 0, writer, accelerator)

    logger.info("Complete")


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse(opt_save_name="lut_export_opt")

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

    output_dir = (
        Path(opt.expDir) / "checkpoints" / f"checkpoint_{opt.startIter}" / "lut"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with accelerator.main_process_first():
        logger_name = "train"
        logger_info(
            logger_name,
            os.path.join(
                Path(opt.expDir)
                / "checkpoints"
                / f"checkpoint_{opt.startIter}"
                / "lut",
                f"export_lut {datetime.datetime.now()} rank={accelerator.process_index}.log",
            ),
        )
        logger = logging.get_logger(logger_name)
        opt_inst.print_options(opt)

    try:
        main(accelerator, opt, logger)
    except BaseException:
        if accelerator.is_main_process:
            raise
