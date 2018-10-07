#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : data_provider.py
#       @date         : 2018/10/07 09:57
import os
import random
from glob import glob
import cv2
import numpy as np

def open_img(path, out_size, gray = False):
    img = cv2.imread(path)
    if out_size is not None:
        img = cv2.resize(img, (out_size, out_size))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.asarray(img) / 127.5 - 1

class DataProvider(object):
    def __init__(self, data_dir, out_size, gray=False):
        self.data_dir = data_dir
        self.file_paths = None
        self.datas = None
        self.out_size = out_size
        self.gray = gray

    def load(self, in_mem = False):
        path_pattern = os.path.join(self.data_dir, "*.jpg")
        self.file_paths = glob(path_pattern)
        if in_mem:
            self.datas = [None] * len(self.file_paths)
            for idx, path in enumerate(self.file_paths):
                self.datas[idx] = open_img(path, self.out_size, self.gray)

    def batchs(self, batch_num, batch_size):
        assert self.file_paths is not None
        for _ in range(batch_num):
            data_idxs = random.sample(range(len(self.file_paths)), batch_size)
            batch = [None] * len(data_idxs)
            if self.datas is None:
                for idx, data_idx in enumerate(data_idxs):
                    batch[idx] = open_img(self.file_paths[data_idx], self.out_size, self.gray)
            else:
                for idx, data_idx in enumerate(data_idxs):
                    batch[idx] = self.datas[data_idxs]
            yield batch

if __name__ == '__main__':
    pass
