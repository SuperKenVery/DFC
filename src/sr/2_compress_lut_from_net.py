import datetime
import warnings

import safetensors
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, logging
from accelerate.utils import ProjectConfiguration

from ..common.config import Experiment, load_experiment
from ..common.utils import logger_info
from ..common.Writer import Logger
from . import model as Model
from .data import SRBenchmark
from .train_utils import get_lut_config, valid_steps

torch.backends.cudnn.benchmark = True
warnings.simplefilter(action="ignore", category=FutureWarning)


def main(accelerator: Accelerator, exp: Experiment, writer, logger):
    config = exp.config

    model_cls = getattr(Model, config.model.model)

    model_G = model_cls(
        sample_size=config.model.sample_size,
        nf=config.model.nf,
        scale=config.model.scale,
        branches=config.model.branches,
        stages=config.model.stages,
    )

    model_G = accelerator.prepare(model_G)
    umodel = accelerator.unwrap_model(model_G)

    # Load saved params
    checkpoint_iter = config.export_lut.checkpoint_iter
    ckpt_dir = exp.get_checkpoint_path(checkpoint_iter)
    accelerator.load_state(str(ckpt_dir))

    lut_cfg = get_lut_config(config.export_lut, config.model.interval)

    lut_ckpt_dir = exp.get_lut_checkpoint_path(checkpoint_iter)
    with umodel.save_as_lut(lut_cfg):
        accelerator.save_model(model_G, str(lut_ckpt_dir))

    # Test exported model
    valid = SRBenchmark(config.data.val_dir, scale=config.model.scale)

    logger.info("Original model before exporting:")
    valid_steps(model_G, valid, exp, 0, writer, accelerator)

    state_dict = safetensors.torch.load_file(str(lut_ckpt_dir / "model.safetensors"))
    with umodel.load_state_from_lut(lut_cfg, accelerator):
        umodel.load_state_dict(state_dict)

    logger.info("Exported model:")
    valid_steps(model_G, valid, exp, 0, writer, accelerator)

    logger.info("Complete")


if __name__ == "__main__":
    exp = load_experiment()
    config = exp.config

    checkpoint_iter = config.export_lut.checkpoint_iter
    lut_output_dir = exp.get_lut_checkpoint_path(checkpoint_iter)
    lut_output_dir.mkdir(parents=True, exist_ok=True)

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
                lut_output_dir
                / f"export_lut_{datetime.datetime.now()}_rank{accelerator.process_index}.log"
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
