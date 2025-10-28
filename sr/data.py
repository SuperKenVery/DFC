import os
import random
import sys
import tarfile
import mmap
import io
import ctypes
import ctypes.util
from accelerate import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")  # run under the project directory
from common.utils import modcrop


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


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [
            str(i).zfill(4) for i in range(1, 901)
        ]  # use both train and valid

        logger = logging.get_logger("train")
        # Use tar archives with memory mapping for optimal random access performance
        self.hr_tar = os.path.join(path, "packed_hr.tar")
        if not os.path.exists(self.hr_tar):
            self.cache_hr()
            logger.info(f"HR images packed to: {self.hr_tar}")
        self.hr_mmap, self.hr_tarfile = self._open_tar_mmap(self.hr_tar)
        logger.info(f"HR tar archive memory-mapped and pre-loaded from: {self.hr_tar}")

        self.lr_tar = os.path.join(path, "packed_lr_x{}.tar".format(self.scale))
        if not os.path.exists(self.lr_tar):
            self.cache_lr()
            logger.info(f"LR images packed to: {self.lr_tar}")
        self.lr_mmap, self.lr_tarfile = self._open_tar_mmap(self.lr_tar)
        logger.info(f"LR tar archive memory-mapped and pre-loaded from: {self.lr_tar}")

    def cache_lr(self):
        """Pack all LR images into a tar archive"""
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        with tarfile.open(self.lr_tar, "w") as tar:
            for f in self.file_list:
                img_path = os.path.join(dataLR, f + "x{}.png".format(self.scale))
                tar.add(img_path, arcname=f + "x{}.png".format(self.scale))

    def cache_hr(self):
        """Pack all HR images into a tar archive"""
        dataHR = os.path.join(self.path, "HR")
        with tarfile.open(self.hr_tar, "w") as tar:
            for f in self.file_list:
                img_path = os.path.join(dataHR, f + ".png")
                tar.add(img_path, arcname=f + ".png")

    def _open_tar_mmap(self, tar_path):
        """Open tar file with memory mapping and lock pages in memory for guaranteed residency"""
        fd = os.open(tar_path, os.O_RDONLY)
        mm = mmap.mmap(
            fd, 0, prot=mmap.PROT_READ, flags=mmap.MAP_POPULATE | mmap.MAP_SHARED
        )

        # Create tarfile object from the memory-mapped file
        tar_file = tarfile.open(fileobj=io.BytesIO(mm))
        return mm, tar_file

    def _get_image_from_tar(self, tarfile_obj, filename):
        """Extract and decode image from tar archive on-demand"""
        member = tarfile_obj.getmember(filename)
        fileobj = tarfile_obj.extractfile(member)
        return np.array(Image.open(fileobj))

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self._get_image_from_tar(self.hr_tarfile, key + ".png")
        im = self._get_image_from_tar(
            self.lr_tarfile, key + "x{}.png".format(self.scale)
        )

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[
            i * self.scale : i * self.scale + self.sz * self.scale,
            j * self.scale : j * self.scale + self.sz * self.scale,
            c,
        ]
        im = im[i : i + self.sz, j : j + self.sz, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)
        im = np.expand_dims(im.astype(np.float32) / 255.0, axis=0)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


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
