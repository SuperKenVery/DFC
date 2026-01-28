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

    # Load saved params
    assert opt.startIter > 0, "Please specify a iter to load"
    ckpt_dir = f"{opt.expDir}/checkpoints/checkpoint_{opt.startIter}"
    accelerator.load_state(ckpt_dir)

    # The LUT module
    lut_model = model(
        sample_size=opt.sample_size,
        nf=opt.nf,
        scale=opt.scale,
        modes=modes,
        stages=stages,
    )
    lut_model = accelerator.prepare(lut_model)
    ulut_model = accelerator.unwrap_model(lut_model)
    lut_cfg = get_lut_cfg(opt)
    state_dict = safetensors.torch.load_file(f"{ckpt_dir}/lut/model.safetensors")  # pyright: ignore[reportAttributeAccessIssue]
    with ulut_model.load_state_from_lut(lut_cfg, accelerator):
        ulut_model.load_state_dict(state_dict)

    logger.info("Debug")
    from remote_pdb import set_trace
    from train_utils import ValidationDataset

    valid = SRBenchmark(opt.valDir, scale=opt.scale)
    val_dataset = ValidationDataset(valid, "Set5", opt.scale)
    im, lb, input_im, key = val_dataset[0]
    im = im.unsqueeze(0)

    model_dbg, lut_dbg = {}, {}
    model_out = model_G(im, debug_info=("", model_dbg))
    lut_out = lut_model(im, debug_info=("", lut_dbg))

    def cmp(key: str, size: int = 5):
        model = model_dbg[key] * 255
        lut = lut_dbg[key] * 255
        diff = torch.abs(lut - model)
        print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")
        return (lut - model)[0, 0, :size, :size]

    print(f"Debug info keys: {lut_dbg.keys()}")

    set_trace()


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
