import os
import cv2
import numpy as np
import torch
from deepvac.aug import Composer, ImageWithMaskAuditFactory, CvAugSegAuditBase4
from deepvac.utils import addUserConfig
from deepvac import LOG

from modules.model import EESPNet_Seg

class ImageWithMaskTargetPositionAudit(CvAugSegAuditBase4):
    def auditConfig(self):
        self.save_dir = self.addUserConfig('target_position_dir', self.config.target_position_dir, "target_position_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        hat_y = np.where(label==1)[0]
        up_y = np.where(label==2)[0]
        down_y = np.where(label==3)[0]
        hat_num = len(hat_y)
        up_num = len(up_y)
        down_num = len(down_y)
        if (hat_num > up_num and up_num!=0) or (hat_num > down_num and down_num!=0):
            LOG.logI("Image {} has risk on rule {} with hat_num = {}, up_num = {} and down_num = {}".format(fn, self.name(), hat_num, up_num, down_num))
            self.write(img, label, fn)
        
        if up_num!=0 and down_num !=0 and np.max(up_y) > np.max(down_y):
            LOG.logI("Image {} has risk on rule {} with up_y = {} and down_y = {}".format(fn, self.name(), np.max(up_y), np.max(down_y)))
            self.write(img, label, fn)
        
        if hat_num!=0 and up_num !=0 and np.max(hat_y) > np.max(up_y):
            LOG.logI("Image {} has risk on rule {} with hat_y = {} and up_y = {}".format(fn, self.name(), np.max(hat_y), np.max(up_y)))
            self.write(img, label, fn)
        return imgs

class ImageWithMaskIouAudit(CvAugSegAuditBase4):
    def auditConfig(self):
        self.model_path = self.addUserConfig('model_path', self.config.model_path, "your model path")
        self.cls_num = self.addUserConfig('cls_num', self.config.cls_num, 4)
        self.device = self.addUserConfig('device', self.config.device, 'cuda')
        self.min_iou = self.addUserConfig('min_iou', self.config.min_iou, 0.4)
        self.save_dir = self.addUserConfig('iou_dir', self.config.target_position_dir, "iou_dir")
        os.makedirs(self.save_dir, exist_ok=True)

        self.net = EESPNet_Seg(self.cls_num).to(self.device)
        self.net.load_state_dict(torch.load(self.model_path))
        self.net = self.net.eval()
        self.composer = self.addUserConfig('composer', self.config.composer, None, True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        sample = self.composer(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_fusion, _, _ = self.net(sample)
            pred_resize = cv2.resize(pred_fusion.squeeze().cpu().numpy().argmax(0), (label.shape[1], label.shape[0]), interpolation = cv2.INTER_NEAREST)
        cls_iou = 0
        for cls_idx in range(1, self.cls_num):
            mask_label = label == cls_idx
            if np.sum(mask_label) == 0:
                continue
            mask_pred = pred_resize == cls_idx
            tmp = np.sum(mask_label & mask_pred)/np.sum(mask_label | mask_pred)
            cls_iou += tmp
        try:
            mean_iou = cls_iou/(len(np.unique(label)) - 1)
        except Exception as e:
            LOG.logE("{} has no annotation yet. Please remove it from dataset.")
            mean_iou = 0

        if mean_iou < self.min_iou:
            LOG.logI("Image {} has risk on rule {} with cls_iou = {}".format(fn, self.name(), mean_iou))
            self.write(img, label, fn)
        return imgs

class ClothesImageWithMaskAuditFactory(ImageWithMaskAuditFactory):
    def initProducts(self):
        super(ClothesImageWithMaskAuditFactory, self).initProducts()
        aug_name_list = ['ImageWithMaskTargetPositionAudit', 'ImageWithMaskIouAudit']
        for aug_name in aug_name_list:
            self.addProduct(aug_name, eval(aug_name))

class ImageWithMaskAuditComposer(Composer):
    def __init__(self, deepvac_config):
        super(ImageWithMaskAuditComposer, self).__init__(deepvac_config)
        af1 = ClothesImageWithMaskAuditFactory('ImageWithMaskIntersectAudit => ImageWithMaskSideRatioAudit => ImageWithMaskTargetPositionAudit => ImageWithMaskTargetSizeAudit => ImageWithMaskIouAudit => ImageWithMaskVisionAudit', deepvac_config)
        #af1 = ClothesImageWithMaskAuditFactory('ImageWithMaskIntersectAudit => ImageWithMaskVisionAudit', deepvac_config)
        self.addAugFactory('af1', af1)

class ImageWithMaskFileLineAuditComposer(Composer):
    def __init__(self, deepvac_config):
        super(ImageWithMaskFileLineAuditComposer, self).__init__(deepvac_config)
        af1 = ClothesImageWithMaskAuditFactory('ImageWithMaskTargetPositionAudit => ImageWithMaskTargetSizeAudit => ImageWithMaskIouAudit => ImageWithMaskVisionAudit', deepvac_config)
        #af1 = ClothesImageWithMaskAuditFactory('ImageWithMaskIntersectAudit => ImageWithMaskVisionAudit', deepvac_config)
        self.addAugFactory('af1', af1)

