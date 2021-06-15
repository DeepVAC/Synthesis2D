import numpy as np
import os
import random
import time
import math
import tqdm

import torch
import cv2

from deepvac import LOG

from network import HRNet

class Synthesis(object):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        self.is_clothes_task = self.config.is_clothes_task
        self.auditConfig()

        self.image_path_list = []
        self.label_path_list = []
        
        self.image_list = []
        self.label_list = []
        
        self.bbox_list = []
        
        self.image_group_list = []
        self.label_group_list = []
        
        self.bbox_group_list = []
        
        self.generated_image_list = []
        self.generated_label_list = []

        if self.is_clothes_task:
            self.human_mask_path_list = []
            self.human_mask_list = []
            self.human_bbox_list = []
            self.human_mask_group_list = []
            self.human_bbox_group_list = []

        self._getImageAndLabelPathList()

        assert len(self.image_path_list)==len(self.label_path_list), "length of image_path_list must be equal to label_path_list."
        if self.is_clothes_task:
            assert len(self.image_path_list)==len(self.human_mask_path_list), "length of human_mask_path_list must be equal to image_path_list."

    def auditConfig(self):
        if not os.path.exists(self.config.output_image_dir):
            os.makedirs(self.config.output_image_dir)
        if not os.path.exists(self.config.output_label_dir):
            os.makedirs(self.config.output_label_dir)

    def _getContours(self, binary_mask):
        binary_mask = np.array(binary_mask, np.uint8)
        ret, binary_mask = cv2.threshold(binary_mask,0,1,cv2.THRESH_BINARY)  
        contours,hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def _clothesMask2Box(self, binary_mask):
        contours = self._getContours(binary_mask)
        ori_h, ori_w = binary_mask.shape
        x, y, x1, y1 = 100000, 100000, -1, -1
        for cnt in contours:
            x_, y_, w_, h_ = cv2.boundingRect(cnt)
            x = min(x, x_)
            y = min(y, y_)
            x1 = max(x1, x_+w_)
            y1 = max(y1, y_+h_)
        
        bounding_box = [max(x, 0), max(y, 0), min(x1, ori_w-1), min(y1, ori_h-1)]
        return bounding_box

    def _personMask2Box(self, binary_mask):
        contours = self._getContours(binary_mask)
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

    def _filterImageForClothes(self):
        self.image_list = []
        self.label_list = []
        self.human_mask_list = []

        for i, (img_path, label_path, human_mask_path) in enumerate(zip(self.image_path_list, self.label_path_list, self.human_mask_path_list)):

            if i % 100 == 0 and i != 0:
                LOG.logI('Filter {} images...'.format(i))

            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            human_mask = cv2.imread(human_mask_path)
            clo_bbox = self._clothesMask2Box(cv2.cvtColor(label, cv2.COLOR_BGR2GRAY))
            person_bbox = self._personMask2Box(cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY))
            if person_bbox is None:
                continue
            src_bbox = [min(clo_bbox[0], person_bbox[0]), min(clo_bbox[1], person_bbox[1]), max(clo_bbox[2], person_bbox[2]), max(clo_bbox[3], person_bbox[3])]
            if src_bbox[2] - src_bbox[0] > 0.5 * img.shape[1]:
                continue
            self.human_bbox_list.append(src_bbox)
            self.human_mask_list.append(human_mask)
            self.bbox_list.append(clo_bbox)
            self.image_list.append(img)
            self.label_list.append(label)
            
    def _filterImageForHuman(self):      
        self.image_list = []
        self.label_list = []

        for i, (img_path, label_path) in enumerate(zip(self.image_path_list, self.label_path_list)):
            if i % 100 == 0 and i != 0:
                LOG.logI('Filter {} images...'.format(i))

            img = cv2.imread(img_path)
            label = cv2.imread(label_path)

            person_bbox = self._personMask2Box(cv2.cvtColor(label, cv2.COLOR_BGR2GRAY))
            if person_bbox is None:
                continue
            if person_bbox[2] - person_bbox[0] > 0.5 * img.shape[1]:
                continue
            self.bbox_list.append(person_bbox)
            self.image_list.append(img)
            self.label_list.append(label)
            
    def _shuffle(self):
        index = [i for i in range(len(self.image_path_list))]
        random.shuffle(index)

        new_image_path_list = []
        new_label_path_list = []
        self.image_path_list, self.label_path_list, self.human_mask_path_list
        if self.is_clothes_task:
            new_human_mask_path_list = []
        for x in index:
            new_image_path_list.append(self.image_path_list[x])
            new_label_path_list.append(self.label_path_list[x])
            if self.is_clothes_task:
                new_human_mask_path_list.append(self.human_mask_path_list[x])
        self.image_path_list, self.label_path_list = new_image_path_list, new_label_path_list
        if self.is_clothes_task:
            self.human_mask_path_list = new_human_mask_path_list


    def _genGroup(self):
        i = 0
        while i < len(self.image_list):
            if len(self.image_list) - i == 1:
                break
            elif len(self.image_list) - i <= 3:
                self.image_group_list.append(self.image_list[i:])
                self.label_group_list.append(self.label_list[i:])
                self.bbox_group_list.append(self.bbox_list[i:])

                if self.is_clothes_task:
                    self.human_mask_group_list.append(self.human_mask_list[i:])
                    self.human_bbox_group_list.append(self.human_bbox_list[i:])
                break

            num = 2 if random.random() < 0.5 else 3

            self.image_group_list.append(self.image_list[i:i+num])
            self.label_group_list.append(self.label_list[i:i+num])
            self.bbox_group_list.append(self.bbox_list[i:i+num])
            
            if self.is_clothes_task:
                self.human_mask_group_list.append(self.human_mask_list[i:i+num])
                self.human_bbox_group_list.append(self.human_bbox_list[i:i+num])

            i += num

    def _processWithThreePerson(self, imgs, labels, bboxes, core_image, core_label, core_bbox, human_masks=None, human_bboxes=None, core_human_box=None):
        for i in range(1, len(imgs)):
            alpha_img = labels[i]
            rgb_img = imgs[i]
            human_img = human_masks[i] if human_masks is not None else None

            xmin, ymin, xmax, ymax = human_bboxes[i] if human_bboxes is not None else bboxes[i]
            alpha_img = alpha_img[ymin:min(ymax, alpha_img.shape[1]), xmin:min(xmax, alpha_img.shape[0])]
            rgb_img = rgb_img[ymin:min(ymax, rgb_img.shape[1]), xmin:min(xmax, rgb_img.shape[0])]

            if human_img is not None:
                human_img = human_img[ymin:min(ymax, human_img.shape[1]), xmin:min(xmax, human_img.shape[0])]

            h, w, _ = rgb_img.shape

            alpha_img = alpha_img.astype('uint8')
            cur_core_box = core_bbox if core_human_box is None else core_human_box
            if i == 1:
                x = cur_core_box[2] + 1
                y = cur_core_box[1] + 1
            else:
                if cur_core_box[0] < w:
                    return None, None
                x = random.randint(1, cur_core_box[0] // 2)
                y = random.randint(1, h // 2)

            dst_mask = core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
            dst_img = core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
                
            if dst_mask.shape[0] == 0 or dst_mask.shape[1] == 0:
                return None, None

            rgb_img = cv2.resize(rgb_img, (dst_img.shape[1], dst_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            alpha_img = cv2.resize(alpha_img, (dst_img.shape[1], dst_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if human_img is not None:
                human_img = cv2.resize(human_img, (dst_img.shape[1], dst_img.shape[0]), interpolation=cv2.INTER_NEAREST) 

            final_mask = np.where(alpha_img != 0, alpha_img, dst_mask)
            final_img = np.where(alpha_img != 0, rgb_img, dst_img)
            if human_img is not None:
                final_img = np.where(human_img != 0, rgb_img, final_img)
         
            core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_img
            core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_mask

        return core_image, core_label

    def _processWithTwoPerson(self, imgs, labels, bboxes, core_image, core_label, core_bbox, right_flag, human_masks=None, human_bboxes=None, core_human_box=None):
        alpha_img = labels[1]
        rgb_img = imgs[1]
        human_img = human_masks[1] if human_masks is not None else None

        xmin, ymin, xmax, ymax = human_bboxes[1] if human_bboxes is not None else bboxes[1]
        alpha_img = alpha_img[ymin:min(ymax, alpha_img.shape[1]), xmin:min(xmax, alpha_img.shape[0])]
        rgb_img = rgb_img[ymin:min(ymax, rgb_img.shape[1]), xmin:min(xmax, rgb_img.shape[0])]
        if human_img is not None:
            human_img = human_img[ymin:min(ymax, human_img.shape[1]), xmin:min(xmax, human_img.shape[0])]

        h, w, _ = rgb_img.shape

        alpha_img = alpha_img.astype('uint8')

        cur_core_box = core_bbox if core_human_box is None else core_human_box
        if cur_core_box[0] < w and not right_flag:
            return None, None

        x = cur_core_box[2] + 1 if right_flag else random.randint(1, cur_core_box[0] // 2)
        y = cur_core_box[1] + 1 if right_flag else random.randint(1, h // 2)

        dst_mask = core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
        dst_img = core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
                
        if dst_mask.shape[0] == 0 or dst_mask.shape[1] == 0:
            return None, None

        rgb_img = cv2.resize(rgb_img, (dst_img.shape[1], dst_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        alpha_img = cv2.resize(alpha_img, (dst_img.shape[1], dst_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if human_img is not None:
            human_img = cv2.resize(human_img, (dst_img.shape[1], dst_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        final_mask = np.where(alpha_img != 0, alpha_img, dst_mask)
        final_img = np.where(alpha_img != 0, rgb_img, dst_img)
        if human_img is not None:
            final_img = np.where(human_img != 0, rgb_img, final_img)
         
        core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_img
        core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_mask

        return core_image, core_label

    def processForClothes(self):
        n = 0
        for imgs, labels, human_masks, bboxes, human_bboxes in zip(self.image_group_list, self.label_group_list, self.human_mask_group_list, self.bbox_group_list, self.human_bbox_group_list):
            core_index = 0
            core_image, core_label, core_bbox, core_human_box = imgs[core_index], labels[core_index], bboxes[core_index], human_bboxes[core_index]
            if core_human_box[2] - core_human_box[0] > core_image.shape[1] *0.5:
                continue
            flag = True
            left = core_human_box[0]
            right = core_image.shape[1] - core_human_box[2]
            right_flag = False
            if right > left:
                right_flag = True

            if len(imgs) == 3:
                core_image, core_label = self._processWithThreePerson(imgs, labels, bboxes, core_image, core_label, core_bbox, human_masks, human_bboxes, core_human_box)
            else:
                core_image, core_label = self._processWithTwoPerson(imgs, labels, bboxes, core_image, core_label, core_bbox, right_flag, human_masks, human_bboxes, core_human_box)
            
            if core_image is None:
                continue
            self.generated_image_list.append(core_image)
            self.generated_label_list.append(core_label)
            cv2.imwrite('{}/{}.jpg'.format(self.config.output_image_dir, n+1), core_image)
            cv2.imwrite('{}/{}.png'.format(self.config.output_label_dir, n+1), core_label)
            n += 1
            if n % 10 == 0:
                LOG.logI('Generate {} images...'.format(n))
    
    def processForHuman(self):
        n = 0
        for imgs, labels, bboxes in zip(self.image_group_list, self.label_group_list, self.bbox_group_list):
            core_index = 0
            core_image, core_label, core_bbox= imgs[core_index], labels[core_index], bboxes[core_index]
            if core_bbox[2] - core_bbox[0] > core_image.shape[1] *0.5:
                continue
            flag = True
            left = core_bbox[0]
            right = core_image.shape[1] - core_bbox[2]
            right_flag = False
            if right > left:
                right_flag = True

            if len(imgs) == 3:
                core_image, core_label = self._processWithThreePerson(imgs, labels, bboxes, core_image, core_label, core_bbox)
            else:
                core_image, core_label = self._processWithTwoPerson(imgs, labels, bboxes, core_image, core_label, core_bbox, right_flag)
            
            if core_image is None:
                continue
            self.generated_image_list.append(core_image)
            self.generated_label_list.append(core_label)
            cv2.imwrite('{}/{}.jpg'.format(self.config.output_image_dir, n+1), core_image)
            cv2.imwrite('{}/{}.png'.format(self.config.output_label_dir, n+1), core_label)
            n += 1
            if n % 10 == 0:
                LOG.logI('Generate {} images...'.format(n))


    def saveImage(self):
        for i in range(len(self.generated_image_list)):
            img, label = self.generated_image_list[i], self.generated_label_list[i]
            cv2.imwrite('{}/{}.jpg'.format(self.config.output_image_dir, i+1), img)
            cv2.imwrite('{}/{}.png'.format(self.config.output_label_dir, i+1), label)


    def _getImageAndLabelPathList(self):
        files = os.listdir(self.config.input_image_dir)
        for i, f in enumerate(files):
            if os.path.isdir(os.path.join(self.config.input_image_dir, f)):
                LOG.logE('{} is a dir, {} contain a sub dir, you must get rid of it.'.format(os.path.join(self.config.input_image_dir, f), self.config.input_image_dir), exit=True)
            if os.path.isdir(os.path.join(self.config.input_label_dir, f.replace('jpg', 'png'))):
                LOG.logE('{} is a dir, {} contain a sub dir, you must get rid of it.'.format(os.path.join(self.config.input_label_dir, f.replace('jpg', 'png')), self.config.input_label_dir), exit=True)
            if self.is_clothes_task and os.path.isdir(os.path.join(self.config.portrait_mask_output_dir, f.replace('jpg', 'png'))):
                LOG.logE('{} is a dir, {} contain a sub dir, you must get rid of it.'.format(os.path.join(self.config.portrait_mask_output_dir, f.replace('jpg', 'png')), self.config.portrait_mask_output_dir), exit=True)
            self.image_path_list.append(os.path.join(self.config.input_image_dir, f))
            self.label_path_list.append(os.path.join(self.config.input_label_dir, f.replace('jpg', 'png')))
            if self.is_clothes_task:
                self.human_mask_path_list.append(os.path.join(self.config.portrait_mask_output_dir, f.replace('jpg', 'png')))
        LOG.logI('Length of image_path_list is {} ...'.format(len(self.image_path_list)))


    def __call__(self):

        self._reset()
        self._getImageAndLabelPathList()

        # LOG.logI('Shuffle begin...')
        # self._shuffle()
        # LOG.logI('Shuffle finish...')

        LOG.logI('Filter images begin...')
        if self.is_clothes_task:
            self._filterImageForClothes()
        else:
            self._filterImageForHuman()
        LOG.logI('Filter images finish...')

        LOG.logI('Generate group begin...')
        self._genGroup()
        LOG.logI('Generate group finish...')

        LOG.logI('Process begin...')
        if self.is_clothes_task:
            self.processForClothes()
        else:
            self.processForHuman()
        LOG.logI('Process finish...')

        # self.saveImage()

class SegUnit(object):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, deepvac_config):
        self.config = deepvac_config
        self.portrait_net = HRNet(num_classes=self.config.num_classes)
        self.portrait_net.load_state_dict(torch.load(self.config.portrait_model))
        self.portrait_net.to(self.config.core.device)
        self.portrait_net.eval()
        os.makedirs(self.config.synthesis.portrait_mask_output_dir, exist_ok=True)

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
    original_img_dir = deepvac_config.synthesis.input_image_dir
    for file_name in tqdm(os.listdir(original_img_dir)):
        file_path = os.path.join(original_img_dir, file_name)
        seg_unit(file_path)

if __name__ == "__main__":
    from config import config as deepvac_config

    #step1, gen portrait mask
    if deepvac_config.use_portrait_mask and deepvac_config.synthesis.is_clothes_task:
        LOG.logI('STEP1: gen portrait mask start')
        genPortraitMask(deepvac_config)
    else:
        LOG.logI('omit STEP1: gen portrait mask start.')

    #step2, synthesis
    LOG.logI('STEP2: synthesis start.')
    synthesis = Synthesis(deepvac_config.synthesis)
    synthesis()