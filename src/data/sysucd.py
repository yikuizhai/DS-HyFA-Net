#Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang, “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection,” IEEE Trans. Geosci. Remote Sensing, vol. 60, pp. 1–16, 2021, doi: 10.1109/TGRS.2021.3085870.


import random
from glob import glob
from os.path import join

import numpy as np

from . import CDDataset


class SYSUCDDataset(CDDataset):
    def __init__(
        self,
        root, phase='train',
        transforms=(None, None, None),
        repeats=1,
        subset='val',
        aug_train=False
    ):
        super().__init__(root, phase, transforms, repeats, subset)
        self.aug_train = aug_train

    def _read_file_paths(self):
        t1_list = sorted(glob(join(self.root, self.subset, 'time1', '**', '*.png'), recursive=True))
        t2_list = sorted(glob(join(self.root, self.subset, 'time2', '**', '*.png'), recursive=True))
        tar_list = sorted(glob(join(self.root, self.subset, 'label', '**', '*.png'), recursive=True))
        assert len(t1_list) == len(t2_list) == len(tar_list)
        return t1_list, t2_list, tar_list

    def fetch_target(self, target_path):
        return (super().fetch_target(target_path)/255).astype(np.bool)

    def preprocess(self, t1, t2, tar):
        if self.phase == 'train' and self.aug_train:
            if random.random() < 0.2:
                # Time reversal
                t1, t2 = t2, t1
            if random.random() < 0.2:
                # Random identity
                t2 = t1
                tar.fill(0)
        return super().preprocess(t1, t2, tar)