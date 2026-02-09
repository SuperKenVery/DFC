import datetime
import warnings

import http_pdb
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

import numpy as np
from PIL import Image

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
    lut_model = model_cls(
        sample_size=config.model.sample_size,
        nf=config.model.nf,
        scale=config.model.scale,
        branches=config.model.branches,
        stages=config.model.stages,
    )

    model_G = accelerator.prepare(model_G)
    umodel = accelerator.unwrap_model(model_G)

    # Load saved params for DNN model
    checkpoint_iter = config.export_lut.checkpoint_iter
    ckpt_dir = exp.get_checkpoint_path(checkpoint_iter)
    accelerator.load_state(str(ckpt_dir))

    # Load LUT checkpoint
    lut_model = accelerator.prepare(lut_model)
    ulut_model = accelerator.unwrap_model(lut_model)
    lut_cfg = get_lut_config(config.export_lut, config.model.interval)
    lut_ckpt_dir = exp.get_lut_checkpoint_path(checkpoint_iter)

    state_dict = safetensors.torch.load_file(str(lut_ckpt_dir / "model.safetensors"))
    with ulut_model.load_state_from_lut(lut_cfg, accelerator):
        ulut_model.load_state_dict(state_dict)

    dnn_output, lut_output, model_dbg, lut_dbg, cmp = run_model('butterfly', model_G, lut_model)

    http_pdb.set_trace()

def run_model(file: str, dnn_model, lut_model, dataset: str = "Set5") :
    test_img_path = config.data.val_dir + f"/{dataset}/LR_bicubic/X4/{file}.png"
    test_img = Image.open(test_img_path).convert("RGB")
    test_img = np.array(test_img).astype(np.float32) / 255.0
    test_tensor = (
        torch.from_numpy(test_img).permute(2, 0, 1).unsqueeze(0).to(accelerator.device)
    )

    # Get output
    model_dbg, lut_dbg = {}, {}

    with torch.no_grad():
        dnn_output = dnn_model(test_tensor, debug_info=("", model_dbg))

    with torch.no_grad():
        lut_output = lut_model(test_tensor, debug_info=("", lut_dbg))

    def cmp(key: str, size: int = 5):
        model = model_dbg[key] * 255
        lut = lut_dbg[key] * 255
        diff = torch.abs(lut - model)
        print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")
        return (lut - model)[0, 0, :size, :size]

    return dnn_output, lut_output, model_dbg, lut_dbg, cmp

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
