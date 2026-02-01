import ctypes
import ctypes.util
import io
import mmap
import os
import random
import sys
import tarfile
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
import torchvision
from accelerate import logging
from numpy import typing as npt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..common.utils import modcrop


class InfiniteDIV2K(Dataset):
    def __init__(self, batch_size, num_workers, scale, path, patch_size):
        self.data = DIV2K(scale, path, patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def __getitem__(self, idx):
        length = len(self.data)
        return self.data[idx % length]


class DIV2K(
    Dataset[
        tuple[
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
        ]
    ]
):
    def __init__(self, scale, path, patch_size):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.path = path
        self.file_list = [
            str(i).zfill(4) for i in range(1, 901)
        ]  # use both train and valid

        logger = logging.get_logger("train")
        # Use tar archives with memory mapping for optimal random access performance
        self.hr_tar = os.path.join(path, "packed_hr_tiff_uncompressed.tar")
        if not os.path.exists(self.hr_tar):
            self.cache_hr()
            logger.info(f"HR images packed to: {self.hr_tar}")
        self.hr_mmap, self.hr_tarfile = self._open_tar_mmap(self.hr_tar)
        logger.info(f"HR tar archive memory-mapped and pre-loaded from: {self.hr_tar}")

        self.lr_tar = os.path.join(
            path, "packed_lr_x{}_tiff_uncompressed.tar".format(self.scale)
        )
        if not os.path.exists(self.lr_tar):
            self.cache_lr()
            logger.info(f"LR images packed to: {self.lr_tar}")
        self.lr_mmap, self.lr_tarfile = self._open_tar_mmap(self.lr_tar)
        logger.info(f"LR tar archive memory-mapped and pre-loaded from: {self.lr_tar}")

    def _convert_image_to_tiff_buffer(self, img) -> tuple[BytesIO, int]:
        """Convert PIL Image to uncompressed TIFF in BytesIO buffer"""

        tiff_buffer = BytesIO()
        img.save(tiff_buffer, compression="none", format="TIFF")
        img_size = tiff_buffer.tell()
        tiff_buffer.seek(0)
        return tiff_buffer, img_size

    def cache_lr(self):
        """Pack all LR images into a tar archive as uncompressed TIFF for fast decoding"""
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))

        with tarfile.open(self.lr_tar, "w") as tar:
            for f in tqdm(self.file_list, dynamic_ncols=True, desc="Caching lr images"):
                # Load PNG image
                img_path = os.path.join(dataLR, f + "x{}.png".format(self.scale))
                img = Image.open(img_path)

                # Convert to uncompressed TIFF in memory
                tiff_buffer, tiff_size = self._convert_image_to_tiff_buffer(img)

                # Create TarInfo and add directly from memory
                tiff_info = tarfile.TarInfo(name=f + "x{}.tiff".format(self.scale))
                tiff_info.size = tiff_size
                tiff_buffer.seek(0)
                tar.addfile(tiff_info, tiff_buffer)

    def cache_hr(self):
        """Pack all HR images into a tar archive as uncompressed TIFF for fast decoding"""
        dataHR = os.path.join(self.path, "HR")

        with tarfile.open(self.hr_tar, "w") as tar:
            for f in tqdm(self.file_list, dynamic_ncols=True, desc="Caching hr images"):
                # Load PNG image
                img_path = os.path.join(dataHR, f + ".png")
                img = Image.open(img_path)

                # Convert to uncompressed TIFF in memory
                tiff_buffer, tiff_size = self._convert_image_to_tiff_buffer(img)

                # Create TarInfo and add directly from memory
                tiff_info = tarfile.TarInfo(name=f + ".tiff")
                tiff_info.size = tiff_size
                tiff_buffer.seek(0)
                tar.addfile(tiff_info, tiff_buffer)

    def _open_tar_mmap(self, tar_path: str):
        """Open tar file with memory mapping and lock pages in memory for guaranteed residency"""
        fd = os.open(tar_path, os.O_RDONLY)
        mm = mmap.mmap(
            fd, 0, prot=mmap.PROT_READ, flags=mmap.MAP_POPULATE | mmap.MAP_SHARED
        )

        # Create tarfile object from the memory-mapped file
        tar_file = tarfile.open(fileobj=io.BytesIO(mm))
        return mm, tar_file

    def _get_image_from_tar(
        self, tarfile_obj: tarfile.TarFile, filename: str
    ) -> torch.Tensor:
        """Extract and decode image from tar archive on-demand"""
        member = tarfile_obj.getmember(filename)
        fileobj = tarfile_obj.extractfile(member)

        # Use PIL Image.open for TIFF support, then convert to numpy array
        img = Image.open(fileobj)
        return torch.tensor(np.array(img)).permute(2, 0, 1)

    def __getitem__(
        self, _dump
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        key = random.choice(self.file_list)
        lb = self._get_image_from_tar(self.hr_tarfile, key + ".tiff")
        im = self._get_image_from_tar(
            self.lr_tarfile, key + "x{}.tiff".format(self.scale)
        )

        shape = im.shape
        i = random.randint(0, shape[1] - self.sz)
        j = random.randint(0, shape[2] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[
            c : c + 1,
            i * self.scale : i * self.scale + self.sz * self.scale,
            j * self.scale : j * self.scale + self.sz * self.scale,
        ]
        im = im[c : c + 1, i : i + self.sz, j : j + self.sz]

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


def rigid_aug(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    im, lb = batch

    if random.uniform(0, 1) < 0.5:
        lb = torch.fliplr(lb)
        im = torch.fliplr(im)

    if random.uniform(0, 1) < 0.5:
        lb = torch.flipud(lb)
        im = torch.flipud(im)

    k = random.choice([0, 1, 2, 3])
    lb = torch.rot90(lb, k=k)
    im = torch.rot90(im, k=k)

    return im / 255.0, lb / 255.0


class SRBenchmark(Dataset):
    def __init__(self, path, scale=4):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = (5 + 14 + 100 + 109 + 100) * 2

        for dataset in ["Set5", "Set14", "B100", "Manga109", "Urban100"]:
            folder = os.path.join(path, dataset, "HR")
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(
                    Image.open(os.path.join(path, dataset, "HR", files[i]))
                )
                im_hr = modcrop(im_hr, scale)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)

                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + "_" + files[i][:-4]
                self.ims[key] = im_hr

                im_lr = np.array(
                    Image.open(
                        os.path.join(
                            path,
                            dataset,
                            "LR_bicubic/X%d" % scale,
                            files[i][:-4] + "x%d.png" % scale,
                        )
                    )
                )  # [:-4] + 'x%d.png'%scale)))
                if len(im_lr.shape) == 2:
                    im_lr = np.expand_dims(im_lr, axis=2)

                    im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                key = dataset + "_" + files[i][:-4] + "x%d" % scale
                self.ims[key] = im_lr

                assert im_lr.shape[0] * scale == im_hr.shape[0]

                assert im_lr.shape[1] * scale == im_hr.shape[1]
                assert im_lr.shape[2] == im_hr.shape[2] == 3

        assert len(self.ims.keys()) == _ims_all
