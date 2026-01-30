import os
import sys

import numpy as np
import torch
from accelerate import logging
from PIL import Image

if ".." not in sys.path:
    sys.path.insert(0, "..")

from common.lut_module import DFCConfig, LUTConfig
from common.utils import PSNR, _rgb2ycbcr

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


def SaveCheckpoint(model_G, opt_G, opt, i, best=False):
    logger = logging.get_logger("train")
    str_best = ""
    if best:
        str_best = "_best"

    torch.save(
        model_G.state_dict(),
        os.path.join(opt.expDir, "Model_{:06d}{}.pth".format(i, str_best)),
    )
    torch.save(opt_G, os.path.join(opt.expDir, "Opt_{:06d}{}.pth".format(i, str_best)))
    logger.info("Checkpoint saved {}".format(str(i)))


def valid_steps(model_G, valid, opt, iter, writer, accelerator):
    logger = logging.get_logger("train")
    if opt.debug:
        datasets = ["Set5", "Set14"]
    else:
        datasets = ["Set5", "Set14"]

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            # Create DataLoader for this dataset
            val_dataset = ValidationDataset(valid, datasets[i], opt.scale)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )

            # Prepare with accelerator for multi-GPU
            val_loader = accelerator.prepare(val_loader)

            psnrs = []

            # Only main process creates directory
            result_path = os.path.join(opt.valoutDir, datasets[i])
            if accelerator.is_main_process:
                if not os.path.isdir(result_path):
                    os.makedirs(result_path)

            for im, lb, input_im, key in val_loader:
                # Remove batch dimension since batch_size=1
                im = im.squeeze(0)
                lb = lb.squeeze(0).cpu().numpy()
                input_im = input_im.squeeze(0).cpu().numpy()
                key = key[0]  # key is a tuple with one element

                # Add batch dimension for model input
                # print(
                #     f"Weight device {model_G.convblock1.DepthwiseBlock0_s.model.lut_weight.device}, "
                #     f"data device {im.device}, "
                #     f"ref2idx device {model_G.convblock1.DepthwiseBlock0_s.model.ref2index.device}, "
                #     f"dfc weight device {model_G.convblock1.DepthwiseBlock0_s.model.diagonal_weight.device}"
                # )
                im = im.unsqueeze(0)
                pred = model_G(im, "valid")
                # pred = accelerator.unwrap_model(model_G).lut_forward(im)

                pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(PSNR(left, right, opt.scale))

                # Only main process saves images
                if accelerator.is_main_process:
                    if iter < 10000:  # save input and gt at start
                        input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(
                            np.uint8
                        )
                        Image.fromarray(input_img).save(
                            os.path.join(
                                result_path, "{}_input.png".format(key.split("_")[-1])
                            )
                        )
                        Image.fromarray(lb.astype(np.uint8)).save(
                            os.path.join(
                                result_path, "{}_gt.png".format(key.split("_")[-1])
                            )
                        )

                    Image.fromarray(pred).save(
                        os.path.join(
                            result_path, "{}_net.png".format(key.split("_")[-1])
                        )
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


def get_lut_cfg(opt):
    if opt.useDFC:
        dfc_cfg = DFCConfig(
            high_precision_interval=opt.interval, diagonal_radius=opt.dw
        )
        lut_cfg = LUTConfig(interval=opt.si, dfc=dfc_cfg)
    else:
        lut_cfg = LUTConfig(interval=opt.interval, dfc=None)
    return lut_cfg
