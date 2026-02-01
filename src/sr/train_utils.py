import os
from pathlib import Path

import numpy as np
import torch
from accelerate import logging
from PIL import Image

from ..common.config import Experiment, ExperimentConfig, ExportLUTConfig
from ..common.lut_module import DFCConfig, LUTConfig
from ..common.utils import PSNR, _rgb2ycbcr

torch.backends.cudnn.benchmark = True

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}


class ValidationDataset(torch.utils.data.Dataset):
    """Wrapper dataset for validation to enable DataLoader usage"""

    def __init__(self, valid, dataset_name, scale):
        self.valid = valid
        self.dataset_name = dataset_name
        self.scale = scale
        self.files = valid.files[dataset_name]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        key = self.dataset_name + "_" + file[:-4]

        lb = self.valid.ims[key]
        input_im = self.valid.ims[key + "x%d" % self.scale]

        input_im = input_im.astype(np.float32) / 255.0
        im = torch.Tensor(np.transpose(input_im, [2, 0, 1]))

        # Return all needed data
        return im, lb, input_im, key


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def valid_steps(
    model_G,
    valid,
    exp: Experiment,
    iter: int,
    writer,
    accelerator,
):
    """Run validation on the model."""
    logger = logging.get_logger("train")
    datasets = ["Set5", "Set14"]
    scale = exp.config.model.scale
    val_output_dir = exp.val_output_dir

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            # Create DataLoader for this dataset
            val_dataset = ValidationDataset(valid, datasets[i], scale)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )

            # Prepare with accelerator for multi-GPU
            val_loader = accelerator.prepare(val_loader)

            psnrs = []

            # Only main process creates directory
            result_path = val_output_dir / datasets[i]
            if accelerator.is_main_process:
                result_path.mkdir(parents=True, exist_ok=True)

            for im, lb, input_im, key in val_loader:
                # Remove batch dimension since batch_size=1
                im = im.squeeze(0)
                lb = lb.squeeze(0).cpu().numpy()
                input_im = input_im.squeeze(0).cpu().numpy()
                key = key[0]  # key is a tuple with one element

                # Add batch dimension for model input
                im = im.unsqueeze(0)
                pred = model_G(im, "valid")

                pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(PSNR(left, right, scale))

                # Only main process saves images
                if accelerator.is_main_process:
                    if iter < 10000:  # save input and gt at start
                        input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(
                            np.uint8
                        )
                        Image.fromarray(input_img).save(
                            result_path / "{}_input.png".format(key.split("_")[-1])
                        )
                        Image.fromarray(lb.astype(np.uint8)).save(
                            result_path / "{}_gt.png".format(key.split("_")[-1])
                        )

                    Image.fromarray(pred).save(
                        result_path / "{}_net.png".format(key.split("_")[-1])
                    )

            # Gather PSNR values from all processes
            psnrs_tensor = torch.tensor(psnrs, device=accelerator.device)
            all_psnrs = accelerator.gather(psnrs_tensor)

            # Only main process logs results
            if accelerator.is_main_process:
                avg_psnr = all_psnrs.cpu().numpy().mean()
                logger.info(
                    "Iter {} | Dataset {} | AVG Val PSNR: {:02f}".format(
                        iter, datasets[i], avg_psnr
                    )
                )
                writer.scalar_summary(
                    "PSNR_valid/{}".format(datasets[i]), avg_psnr, iter
                )


def get_lut_config(export_config: ExportLUTConfig, model_interval: int) -> LUTConfig:
    """
    Create LUTConfig from export configuration.

    Args:
        export_config: The export_lut section of ExperimentConfig
        model_interval: The model.interval value (used when DFC is disabled)
    """
    if export_config.dfc.enabled:
        dfc_cfg = DFCConfig(
            high_precision_interval=model_interval,
            diagonal_radius=export_config.dfc.diagonal_width,
        )
        lut_cfg = LUTConfig(interval=export_config.dfc.sampling_interval, dfc=dfc_cfg)
    else:
        lut_cfg = LUTConfig(interval=model_interval, dfc=None)
    return lut_cfg
