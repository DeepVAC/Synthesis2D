# -*- coding:utf-8 -*-
import sys
import os
import glob

import cv2
from tqdm import tqdm
import torch

from deepvac import LOG
from deepvac.datasets import CocoCVSegDataset

if __name__ == "__main__":
    from config import config
    json_path_glob = sys.argv[1] + "/*.json"

    json_paths = glob.glob(json_path_glob)
    sample_path_prefixs = [os.path.splitext(jp)[0] for jp in json_paths]
    LOG.logI("All json_paths: {} \n All sample_path_prefixs: {}".format(json_paths, sample_path_prefixs))

    for sample_path_prefix, json_path in zip(sample_path_prefixs, json_paths):
        if not os.path.exists(sample_path_prefix):
            LOG.logE("Path {} not exists !".format(sample_path_prefix), exit=True)
        config.test_dataset = CocoCVSegDataset(config, sample_path_prefix, json_path, config.cat2idx)
        config.test_loader = torch.utils.data.DataLoader(config.test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        for idx, (img, label, _, file_path) in tqdm(enumerate(config.test_loader), total=config.test_loader.__len__()):
            continue
        LOG.logI("Json file {} analyze done!".format(json_path))
