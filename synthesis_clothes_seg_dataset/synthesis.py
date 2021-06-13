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

def src_mask_to_box(binary_mask):
    contours = get_contours(binary_mask)
    img_h, img_w = binary_mask.shape
    min_x = 100000
    min_y = 100000
    max_x = -1
    max_y = -1
    for cnt in contours:
        x, y, w, h  = cv2.boundingRect(cnt)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x+w)
        max_y = max(max_y, y+h )

    bounding_box = [max(min_x, 0), max(min_y, 0), min(max_x, img_w-1), min(max_y, img_h-1)]
    return bounding_box

def portrait_mask_to_box(binary_mask):
    contours = get_contours(binary_mask)
    img_h, img_w = binary_mask.shape
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    if len(areas) == 0:
        return None
    idx = areas.index(np.max(areas))
    x, y, w, h = cv2.boundingRect(contours[idx])
    bounding_box = [max(x, 0), max(y, 0), min(x+w, img_w-1), min(y+h, img_h-1)]
    return bounding_box

def paste(src_img, dst_img, src_mask, portrait_mask=None):
    original_src_bbox = src_mask_to_box(cv2.cvtColor(src_mask, cv2.COLOR_BGR2GRAY))
    portrait_bbox = None
    #if use portrait to help
    if portrait_mask is None:
        src_bbox = original_src_bbox
        final_mask = src_mask
    else:
        portrait_bbox = portrait_mask_to_box(cv2.cvtColor(portrait_mask, cv2.COLOR_BGR2GRAY))
        if portrait_bbox is None:
            return None, None
        src_bbox = [min(original_src_bbox[0], portrait_bbox[0]), min(original_src_bbox[1], portrait_bbox[1]), max(original_src_bbox[2], portrait_bbox[2]), max(original_src_bbox[3], portrait_bbox[3])]
        final_mask = np.where(portrait_mask != 0, portrait_mask, src_mask)

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

    if scale_src_bbox_w * scale_src_bbox_h < h * w / 100:
        LOG.logI('Target person is too small for dst image: {} vs {}'.format(scale_src_bbox_w * scale_src_bbox_h, h * w))
        return None, None

    src_img_crop = src_img[src_bbox[1]:src_bbox[1]+src_bbox_h, src_bbox[0]:src_bbox[0]+src_bbox_w, :]
    src_mask_crop = src_mask[src_bbox[1]:src_bbox[1]+src_bbox_h, src_bbox[0]:src_bbox[0]+src_bbox_w, :]
    final_mask_crop = final_mask[src_bbox[1]:src_bbox[1]+src_bbox_h, src_bbox[0]:src_bbox[0]+src_bbox_w, :]
    src_img_crop = cv2.resize(src_img_crop, (scale_src_bbox_w, scale_src_bbox_h), interpolation=cv2.INTER_NEAREST)
    src_mask_crop = cv2.resize(src_mask_crop, (scale_src_bbox_w, scale_src_bbox_h), interpolation=cv2.INTER_NEAREST)
    final_mask_crop = cv2.resize(final_mask_crop, (scale_src_bbox_w, scale_src_bbox_h), interpolation=cv2.INTER_NEAREST)

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
        if deepvac_config.use_portrait_mask:
            portrait_mask = cv2.imread(os.path.join(deepvac_config.portrait_mask_output_dir, os.path.splitext(name)[0]+'.png'))
        else:
            portrait_mask = None
            
        for idx in range(deepvac_config.multiple):
            src_img_copy = src_img.copy()
            src_mask_copy = src_mask.copy()
            portrait_mask_copy = portrait_mask.copy() if portrait_mask is not None else None
            index = random.randint(0, bk_imgs_length-1)
            dst_img = cv2.imread(os.path.join(deepvac_config.bg_dir, bk_imgs[index]))
            result, mask = paste(src_img_copy, dst_img, src_mask_copy, portrait_mask_copy)
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
    if deepvac_config.use_portrait_mask:
        LOG.logI('STEP1: gen portrait mask start')
        genPortraitMask(deepvac_config)
    else:
        LOG.logI('omit STEP1: gen portrait mask start.')

    #step2, synthesis
    LOG.logI('STEP2: synthesis start.')
    synthesis(deepvac_config)

    # step3, show mask
    LOG.logI('STEP3: show result start.')
    showMask(deepvac_config)

