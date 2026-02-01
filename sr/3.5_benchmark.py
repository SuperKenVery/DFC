import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

import model as Model
from data import Provider, SRBenchmark

sys.path.insert(0, "../")  # run under the current directory
from common.option import TrainOptions
from common.utils import PSNR, cal_ssim, logger_info, _rgb2ycbcr

torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    logger_name = 'lutft'
    logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, opt.model)

    input_im = torch.rand(opt.batchSize, 1, opt.cropSize, opt.cropSize).cuda()

    model_G = model(lut_folder=opt.expDir, modes=modes, stages=stages, lutName=opt.load_lutName, upscale=opt.scale, interval=opt.interval,
                    compressed_dimensions=opt.cd, diagonal_width=opt.dw, sampling_interval=opt.si, sample_size=opt.sample_size, phase='not train').cuda()
    model_G.train(False)

    with torch.no_grad():
        for i in tqdm(range(opt.startIter + 1, opt.totalIter + 1), dynamic_ncols=True):
            pred = model_G(input_im)

    logger.info("Complete")
