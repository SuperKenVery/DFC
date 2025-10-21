import torch
import cv2
import numpy as np
from scipy import signal
from accelerate import logging
import logging as pylogging
from tqdm import tqdm
from tqdm.utils import _screen_shape_wrapper
import time, sys
from typing import Mapping, List, Tuple, Union


def logger_info(logger_name, log_path='default_logger.log'):
    log = logging.get_logger(logger_name)
    log.logger.handlers.clear()
    log.logger.propagate = False

    formatter = pylogging.Formatter(
        '%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    fh = pylogging.FileHandler(log_path, mode='a')
    fh.setFormatter(formatter)
    log.setLevel("INFO")
    log.logger.addHandler(fh)

    sh = pylogging.StreamHandler()
    sh.setFormatter(formatter)
    log.logger.addHandler(sh)

def modcrop(image, modulo):
    if len(image.shape) == 2:
        sz = image.shape
        sz = sz - np.mod(sz, modulo)
        image = image[0:sz[0], 0:sz[1]]
    elif image.shape[2] == 3:
        sz = image.shape[0:2]
        sz = sz - np.mod(sz, modulo)
        image = image[0:sz[0], 0:sz[1], :]
    else:
        raise NotImplementedError
    return image


def _rgb2ycbcr(img, maxVal=255):
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def PSNR(y_true, y_pred, shave_border=4):
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255. / rmse)


def cal_ssim(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    M, N = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim

def cal_ssim_rgb(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    M, N, num_channels = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    ssim_values = []
    for channel in range(num_channels):
        img1_channel = np.float64(img1[:, :, channel])
        img2_channel = np.float64(img2[:, :, channel])

        mu1 = signal.convolve2d(img1_channel, window, 'valid')
        mu2 = signal.convolve2d(img2_channel, window, 'valid')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = signal.convolve2d(img1_channel * img1_channel, window, 'valid') - mu1_sq
        sigma2_sq = signal.convolve2d(img2_channel * img2_channel, window, 'valid') - mu2_sq
        sigma12 = signal.convolve2d(img1_channel * img2_channel, window, 'valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        mssim = np.mean(ssim_map)
        ssim_values.append(mssim)

    avg_ssim = np.mean(ssim_values)
    return avg_ssim

def cPSNR(y_true, y_pred, shave_border=0):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border, :]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255. / rmse)

def bPSNR(img1, img2, crop_border):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :], img2[:, c:c + 1, :, :], reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]

def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef

class multiline_tqdm(tqdm):

    def __init__(self, *args, newline_thres: float=0.7, is_subbar: bool=False, desc="", **kwargs):

        super().__init__(*args, **kwargs)
        self.subbar = None
        self.newline_thres = newline_thres
        self.is_subbar = is_subbar
        self.set_description(desc)
        self.kwargs = kwargs

    def set_description(self, desc="", refresh=True):
        screen_width, _ = _screen_shape_wrapper()(sys.stdout)
        max_len = screen_width
        if len(desc) > max_len*self.newline_thres:
            self.ensure_subbar()
            super().set_description_str(desc=desc[:screen_width], refresh=refresh)
            self.subbar.set_description(desc[screen_width:])
        else:
            self.clear_subbar()
            super().set_description(desc=desc, refresh=refresh)

    def set_postfix(self, ordered_dict: Mapping[str, object] | None = None, refresh: bool | None = True, **kwargs):
        if not self.subbar:
            self.ensure_subbar()
        super().set_postfix(ordered_dict, refresh, **kwargs)

    def ensure_subbar(self):
        if not self.subbar:
            self.subbar = multiline_tqdm(range(len(self)), is_subbar=True, **self.kwargs)
            self.subbar.n = self.n
            self.default_bar_format = self.bar_format
            self.bar_format = "{desc} {postfix}"

    def clear_subbar(self):
        if self.subbar:
            self.bar_format = self.default_bar_format
            self.subbar.leave = False
            self.subbar.close()
            self.subbar = None

    def update(self, n=1):
        if self.subbar:
            self.subbar.update(n)
            self.last_print_n = self.subbar.last_print_n
            self.n = self.subbar.n
        else:
            super().update(n)

    def close(self):
        if self.subbar:
            self.subbar.leave = self.leave
            self.subbar.close()

        super().close()

if __name__=='__main__':
    bar = multiline_tqdm(range(100), dynamic_ncols=True)
    # bar.set_description("progress barrr")
    for i in bar:
        bar.set_postfix({'i': i, '10i': 10*i, '100i': 100*i})
        time.sleep(1)
