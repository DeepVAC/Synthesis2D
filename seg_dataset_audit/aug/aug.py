import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

import torch
from torchvision import transforms as trans

from deepvac.aug.factory import AugFactory
from deepvac.aug import CvAugBase, Composer
from deepvac.utils import addUserConfig

from modules.model import EESPNet_Seg

class CvAugBase4(CvAugBase):
    def __init__(self, deepvac_config):
        super(CvAugBase4, self).__init__(deepvac_config)
        self.input_len = self.addUserConfig('input_len', self.config.input_len, 4)
        self.cls_num = self.addUserConfig('cls_num', self.config.cls_num, 4)
        self.pallete = [[255, 255, 255],
            [255, 0,  0],
            [0, 255,  0],
            [0,  0,  255],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70,  130, 180],
            [220, 20,  60],
            [255, 0,   0],
            [0,   0,   142],
            [0,   0,   70],
            [0,   60,  100],
            [0,   80,  100],
            [0,   0,   230],
            [119, 11,  32]]

    def putMask(self, img, mask):
        h,w = img.shape[:2]
        classMap_numpy_color = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(self.cls_num):
            [r, g, b] = self.pallete[idx]
            classMap_numpy_color[mask == idx] = [b, g, r]
        overlayed = cv2.addWeighted(img, 0.5, classMap_numpy_color, 0.5, 0)
        return overlayed

class ImageWithMaskIntersectAudit(CvAugBase4):
    def auditConfig(self):
        self.background_index = self.addUserConfig('background_index', self.config.background_index, 1)
        self.intersect_ratio = self.addUserConfig('intersect_ratio', self.config.intersect_ratio, 0.10)
        self.save_dir = self.addUserConfig('intersect_dir', self.config.intersect_dir, "intersect_dir")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def forward(self, imgs):
        img, label, cls_masks, fn = imgs
        len_mask = len(cls_masks)
        if len_mask < 2:
            return imgs
        for i in range(len_mask - 1):
            for j in range(i+1, len_mask):
                mask1 = cls_masks[i]
                mask2 = cls_masks[j]
                min_area = min(len(np.where(mask1==1)[0]), len(np.where(mask2==1)[0]))
                if len(np.where(mask1*mask2==1)[0]) / min_area >= self.intersect_ratio:
                    overlayed = self.putMask(img, label)
                    cv2.imwrite(os.path.join(self.save_dir, fn), overlayed)
        return imgs

class ImageWithMaskSideRatioAudit(CvAugBase4):
    def auditConfig(self):
        self.save_dir = self.addUserConfig('side_ratio_dir', self.config.side_ratio_dir, "side_ratio_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        if img.shape[0] == img.shape[1]:
            overlayed = self.putMask(img, label)
            cv2.imwrite(os.path.join(self.save_dir, fn), overlayed)
        return imgs

class ImageWithMaskTargetPositionAudit(CvAugBase4):
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
        if (hat_num > up_num and up_num!=0) or (hat_num > down_num and down_num!=0) or \
            up_num!=0 and down_num !=0 and np.max(up_y) > np.max(down_y):
            overlayed = self.putMask(img, label)
            cv2.imwrite(os.path.join(self.save_dir, fn), overlayed)
        return imgs

class ImageWithMaskTargetSizeAudit(CvAugBase4):
    def auditConfig(self):
        self.min_ratio = self.addUserConfig('min_ratio', self.config.min_ratio, 900)
        self.save_dir = self.addUserConfig('target_position_dir', self.config.target_position_dir, "target_size_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        hat_y = np.where(label==1)[0]
        up_y = np.where(label==2)[0]
        down_y = np.where(label==3)[0]
        hat_num = len(hat_y)
        up_num = len(up_y)
        down_num = len(down_y)
        benchmark = int(label.shape[0] * label.shape[1] / self.min_ratio)
        if (hat_num < benchmark and hat_num > 0) or (up_num < benchmark and up_num > 0) or (down_num < benchmark and down_num > 0):
            overlayed = self.putMask(img, label)
            cv2.imwrite(os.path.join(self.save_dir, fn), overlayed)
        return imgs

class ImageWithMaskIouAudit(CvAugBase4):
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

        self.composer = trans.Compose([trans.ToPILImage(),
            trans.Resize((384, 384)),
            trans.ToTensor(),
            trans.Normalize(mean=torch.Tensor([137.78314, 141.16818, 149.62434]) / 255., std=torch.Tensor([63.96097 , 64.199165, 64.6029]))])

    def forward(self, imgs):
        img, label, _, fn = imgs
        sample = self.composer(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_fusion, _, _ = self.net(sample)
            pred_resize = cv2.resize(pred_fusion.to('cpu').numpy().max(1).squeeze(), (label.shape[1], label.shape[0]), interpolation = cv2.INTER_NEAREST)

        cls_iou = 0
        for cls_idx in range(self.cls_num):
            if cls_idx == 0:
                continue
            mask_label = label == cls_idx
            mask_pred = pred_resize == cls_idx
            cls_iou += np.sum(mask_label & mask_pred)/np.sum(mask_label | mask_pred)
        if cls_iou < self.min_iou:
            overlayed = self.putMask(img, label)
            cv2.imwrite(os.path.join(self.save_dir, fn), overlayed)
        return imgs

class ImageWithMaskVisionAudit(CvAugBase4):
    def auditConfig(self):
        self.save_dir = self.addUserConfig('vision_dir', self.config.vision_dir, "vision_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        overlayed = self.putMask(img, label)
        cv2.imwrite(os.path.join(self.save_dir, fn), overlayed)
        return imgs

class CheckFactory(AugFactory):
    def initProducts(self):
        super(CheckFactory, self).initProducts()
        aug_name_list = ['ImageWithMaskIntersectAudit', 'ImageWithMaskSideRatioAudit', 'ImageWithMaskTargetPositionAudit', 'ImageWithMaskTargetSizeAudit', 'ImageWithMaskIouAudit', 'ImageWithMaskVisionAudit']
        for aug_name in aug_name_list:
            self.addProduct(aug_name, eval(aug_name))

class CheckComposer(Composer):
    def __init__(self, deepvac_config):
        super(CheckComposer, self).__init__(deepvac_config)
        ac1 = CheckFactory('ImageWithMaskVisionAudit => ImageWithMaskIntersectAudit => ImageWithMaskSideRatioAudit => ImageWithMaskTargetPositionAudit => ImageWithMaskTargetSizeAudit => ImageWithMaskIouAudit', deepvac_config)
        self.addAugFactory('ac1', ac1)
