import torch
from torchvision import transforms as trans
from deepvac import config, AttrDict
from deepvac.datasets import CocoCVSegDataset, FileLineCvSegAuditDataset
from aug.aug import *
#aug
config.aug.ImageWithMaskIntersectAudit = AttrDict()
config.aug.ImageWithMaskIntersectAudit.is_consider_bg = True
config.aug.ImageWithMaskIntersectAudit.intersect_ratio = 0.1
config.aug.ImageWithMaskIntersectAudit.remask = True

config.aug.ImageWithMaskTargetSizeAudit = AttrDict()
# any targets must gt 1/900 of original images
config.aug.ImageWithMaskTargetSizeAudit.min_ratio = 900

config.aug.ImageWithMaskIouAudit = AttrDict()
config.aug.ImageWithMaskIouAudit.model_path = 'output/LTS_b1_semantic_detail_fusion_1/model__2021-06-08-18-46__acc_0__epoch_90__step_1245__lr_3.7767762e-05.pth'
config.aug.ImageWithMaskIouAudit.cls_num = 4
config.aug.ImageWithMaskIouAudit.min_iou = 0.4
config.aug.ImageWithMaskIouAudit.composer = trans.Compose([trans.ToPILImage(),
    trans.Resize((384, 384)),
    trans.ToTensor(),
    trans.Normalize(mean=torch.Tensor([137.78314, 141.16818, 149.62434]) / 255., std=torch.Tensor([63.96097 , 64.199165, 64.6029]))])

#dataset
config.datasets.CocoCVSegDataset = AttrDict()
config.datasets.CocoCVSegDataset.composer = ImageWithMaskAuditComposer(config)
config.datasets.CocoCVSegDataset.auto_detect_subdir_with_basenum = 4
config.cat2idx = {1:0, 25:1, 22:2, 59:2, 23:3, 60:0}


# dataset
config.fileline_path = ''
config.delimiter = ','
config.sample_path_prefix = ''
config.datasets.FileLineCvSegAuditDataset = AttrDict()
config.datasets.FileLineCvSegAuditDataset.composer = ImageWithMaskFileLineAuditComposer(config)
