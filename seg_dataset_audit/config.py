import torch
from deepvac import config, AttrDict
from deepvac.datasets import CocoCVSegDataset
from aug.aug import *

config.core.cat2idx = {1:0, 25:1, 22:2, 59:2, 23:3}
config.core.test_sample_path = "your sample path"
config.core.test_target_path = "your coco json path"

config.datasets.CocoCVSegDataset = AttrDict()
config.datasets.CocoCVSegDataset.composer = CheckComposer(config)
config.core.test_dataset = CocoCVSegDataset(config, config.core.test_sample_path, config.core.test_target_path, config.core.cat2idx)
config.core.test_loader = torch.utils.data.DataLoader(config.core.test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
