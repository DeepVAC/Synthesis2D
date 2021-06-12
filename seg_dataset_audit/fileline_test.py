# -*- coding:utf-8 -*-
import sys
import os
import glob

import cv2
from tqdm import tqdm
import torch

from deepvac import LOG
from deepvac.datasets import CocoCVSegDataset, FileLineCvSegAuditDataset

if __name__ == "__main__":
    from config import config
    if len(sys.argv) >=2:
        config.file_path = sys.argv[1]
    
    if len(sys.argv) >=3:
        config.sample_path_prefix = sys.argv[2]

    config.core.test_dataset = FileLineCvSegAuditDataset(config, fileline_path=config.file_path, delimiter=config.delimiter, sample_path_prefix=config.sample_path_prefix)
    config.core.test_loader = torch.utils.data.DataLoader(config.core.test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    for idx, (img, label, _, file_path) in tqdm(enumerate(config.core.test_loader), total=config.core.test_loader.__len__()):
        pass

    LOG.logI("fileline file {} analyze done!".format(config.file_path))


