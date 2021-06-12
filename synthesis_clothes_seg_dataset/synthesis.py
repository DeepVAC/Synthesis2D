import os
import sys
import random
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.nn import functional as F
from network import HRNet
from deepvac import LOG
from deepvac.utils import getOverlayFromSegMask

def get_contours(binary_mask):
    binary_mask = np.array(binary_mask, np.uint8)
    ret, binary_mask = cv2.threshold(binary_mask,0,1,cv2.THRESH_BINARY)  
    contours,hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def clothes_mask_to_box(binary_mask):
    contours = get_contours(binary_mask)
    ori_h, ori_w = binary_mask.shape
    x = 100000
    y = 100000
    x1 = -1
    y1 = -1
    for cnt in contours:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        x = min(x, x_)
        y = min(y, y_)
        x1 = max(x1, x_+w_)
        y1 = max(y1, y_+h_)
        
    bounding_box = [max(x, 0), max(y, 0), min(x1, ori_w-1), min(y1, ori_h-1)]
    return bounding_box

def person_mask_to_box(binary_mask):
    contours = get_contours(binary_mask)
    ori_h, ori_w = binary_mask.shape
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    if len(areas) == 0:
        return None
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bounding_box = [max(x, 0), max(y, 0), min(x+w, ori_w-1), min(y+h, ori_h-1)]
    return bounding_box

def paste(source_img, clothes_mask, person_mask, dst_img):
    clo_bbox = clothes_mask_to_box(cv2.cvtColor(clothes_mask, cv2.COLOR_BGR2GRAY))
    person_bbox = person_mask_to_box(cv2.cvtColor(person_mask, cv2.COLOR_BGR2GRAY))
    if person_bbox is None:
        return None, None

    src_bbox = [min(clo_bbox[0], person_bbox[0]), min(clo_bbox[1], person_bbox[1]), max(clo_bbox[2], person_bbox[2]), max(clo_bbox[3], person_bbox[3])]
    
    src_bbox_h, src_bbox_w = src_bbox[3] - src_bbox[1], src_bbox[2] - src_bbox[0]
    h, w, _ = dst_img.shape
    dst_mask = np.zeros((h, w, 3))
    x = random.randint(1, w // 2)
    y = random.randint(1, h // 2)
    #random scale the human
    scale = random.random() * 2
    scale = scale if scale > 0.5 else 0.5
    scale_src_bbox_w = int(src_bbox_w * scale)
    scale_src_bbox_h = int(src_bbox_h * scale)
    scale_src_bbox_w = scale_src_bbox_w if x + scale_src_bbox_w < w else (w - 1 - x)
    scale_src_bbox_h = scale_src_bbox_h if y + scale_src_bbox_h < h else (h - 1 - y)

    #merge person mask and clothes mask
    final_mask = np.where(person_mask != 0, person_mask, clothes_mask)

    src_img_crop = source_img[src_bbox[1]:src_bbox[1]+src_bbox_h, src_bbox[0]:src_bbox[0]+src_bbox_w, :]
    src_mask_crop = clothes_mask[src_bbox[1]:src_bbox[1]+src_bbox_h, src_bbox[0]:src_bbox[0]+src_bbox_w, :]
    final_mask_crop = final_mask[src_bbox[1]:src_bbox[1]+src_bbox_h, src_bbox[0]:src_bbox[0]+src_bbox_w, :]
    src_img_crop = cv2.resize(src_img_crop, (scale_src_bbox_w, scale_src_bbox_h))
    src_mask_crop = cv2.resize(src_mask_crop, (scale_src_bbox_w, scale_src_bbox_h))
    final_mask_crop = cv2.resize(final_mask_crop, (scale_src_bbox_w, scale_src_bbox_h))

    #paste the src img to dst img
    dst_img_crop = dst_img[y:y+scale_src_bbox_h, x:x+scale_src_bbox_w, :]
    result_img = np.where(final_mask_crop != 0, src_img_crop, dst_img_crop)
    dst_img[y:y+scale_src_bbox_h, x:x+scale_src_bbox_w, :] = result_img
    dst_mask[y:y+scale_src_bbox_h, x:x+scale_src_bbox_w, :] = src_mask_crop
    return dst_img, dst_mask

class SegUnit(object):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, deepvac_config):
        self.config = deepvac_config
        self.portrait_net = HRNet(num_classes=self.config.num_classes)
        self.portrait_net.load_state_dict(torch.load(self.config.portrait_model))
        self.portrait_net.to(self.config.core.device)
        self.portrait_net.eval()
        os.makedirs(self.config.portrait_mask_output_dir, exist_ok=True)

    def transform(self, img_file):
        # ndarray
        image = cv2.imread(img_file, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.config.input_size)
        image = (image / 255. - self.mean) / self.std
        # torch tensor
        img = torch.from_numpy(image).float()
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        return img.to(self.config.core.device)

    def __call__(self, img_file):
        image = cv2.imread(img_file)
        h, w = image.shape[:2]
        image0 = image1 = cv2.resize(image, (448, 448))
        img = self.transform(img_file)

        with torch.no_grad():
            portrait_mask = self.portrait_net({"images": img})["pred"]
        portrait_mask = np.argmax(portrait_mask.cpu().numpy(), axis=1)[0]
        mask = cv2.resize(portrait_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        fn = os.path.splitext(os.path.basename(img_file))[0]+'.png'
        cv2.imwrite(os.path.join(self.config.portrait_mask_output_dir, fn), mask)

def genPortraitMask(deepvac_config):
    seg_unit = SegUnit(deepvac_config)
    original_img_dir = os.path.join(deepvac_config.original_image_label_dir, 'images')
    for file_name in tqdm(os.listdir(original_img_dir)):
        file_path = os.path.join(original_img_dir, file_name)
        seg_unit(file_path)

def synthesis(deepvac_config):
    synthesis_output_img_dir = os.path.join(deepvac_config.synthesis_output_dir, "images")
    synthesis_output_label_dir = os.path.join(deepvac_config.synthesis_output_dir, "labels")

    if not os.path.exists(synthesis_output_img_dir):
        os.makedirs(synthesis_output_img_dir)
    
    if not os.path.exists(synthesis_output_label_dir):
        os.makedirs(synthesis_output_label_dir)

    img_names = os.listdir(os.path.join(deepvac_config.original_image_label_dir, "images"))
    bk_imgs = os.listdir(deepvac_config.bg_dir)
    bk_imgs_length = len(bk_imgs)
    count = 0
    for name in tqdm(img_names):
        src_img = cv2.imread(os.path.join(deepvac_config.original_image_label_dir, "images", name))
        src_mask = cv2.imread(os.path.join(deepvac_config.original_image_label_dir, "labels", os.path.splitext(name)[0]+'.png'))
        person_mask = cv2.imread(os.path.join(deepvac_config.portrait_mask_output_dir, os.path.splitext(name)[0]+'.png'))
        for idx in range(deepvac_config.multiple):
            src_img_copy = src_img.copy()
            src_mask_copy = src_mask.copy()
            person_mask_copy = person_mask.copy()
            index = random.randint(0, bk_imgs_length-1)
            dst_img = cv2.imread(os.path.join(deepvac_config.bg_dir, bk_imgs[index]))
            result, mask = paste(src_img_copy, src_mask_copy, person_mask_copy, dst_img)
            if result is None:
                continue
            cv2.imwrite(os.path.join(synthesis_output_img_dir, 'deepvac_synthesis_{}.jpg'.format(count)), result)
            cv2.imwrite(os.path.join(synthesis_output_label_dir, 'deepvac_synthesis_{}.png'.format(count)), mask)
            count += 1

def showMask(deepvac_config):
    synthesis_output_img_dir = os.path.join(deepvac_config.synthesis_output_dir, "images")
    synthesis_output_label_dir = os.path.join(deepvac_config.synthesis_output_dir, "labels")
    synthesis_output_show_dir = os.path.join(deepvac_config.synthesis_output_dir, "show")
    os.makedirs(synthesis_output_show_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(synthesis_output_img_dir)):
        img = cv2.imread(os.path.join(synthesis_output_img_dir, img_name))
        mask = cv2.imread(os.path.join(synthesis_output_label_dir, os.path.splitext(img_name)[0]+'.png'))[:,:,0]
        overlayed = getOverlayFromSegMask(img, mask)
        cv2.imwrite(os.path.join(synthesis_output_show_dir, img_name), overlayed)

if __name__ == "__main__":
    from config import config as deepvac_config

    #original image dir
    if len(sys.argv) >= 2:
        deepvac_config.original_image_label_dir = sys.argv[1]

    if len(sys.argv) >= 3:
        deepvac_config.synthesis_output_dir = sys.argv[2]

    #step1, gen portrait mask
    LOG.logI('STEP1: gen portrait mask start')
    genPortraitMask(deepvac_config)

    #step2, synthesis
    LOG.logI('STEP2: synthesis start')
    synthesis(deepvac_config)

    # step3, show mask
    LOG.logI('STEP3: show result start')
    showMask(deepvac_config)
