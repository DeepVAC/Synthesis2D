# -*- coding:utf-8 -*-
import sys
import os
import time
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import deepvac
from deepvac import LOG, Deepvac
from deepvac.datasets import CocoCVSegDataset

if __name__ == "__main__":
    from config import config

    for idx, (img, label) in tqdm(enumerate(config.core.test_loader), total=config.core.test_loader.__len__()):
        continue
