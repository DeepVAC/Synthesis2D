import numpy as np
import os
import random

import cv2

from deepvac import LOG

class Synthesis(object):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        self.auditConfig()

        self.image_path_list = []
        self.label_path_list = []
        self.human_mask_path_list = []

        self.image_list = []
        self.label_list = []
        self.human_mask_list = []

        self.bbox_list = []
        self.human_bbox_list = []

        self.image_group_list = []
        self.label_group_list = []
        self.human_mask_group_list = []

        self.bbox_group_list = []
        self.human_bbox_group_list = []

        self.generated_image_list = []
        self.generated_label_list = []

        self._getImageAndLabelPathList()

        assert len(self.image_path_list)==len(self.label_path_list) and len(self.image_path_list)==len(self.human_mask_path_list), "length of image_path_list must be equal to label_path_list and human_mask_path_list"

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

    def _filterImage(self):
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

            self.bbox_list.append(clo_bbox)
            self.human_bbox_list.append(src_bbox)
            self.image_list.append(img)
            self.label_list.append(label)
            self.human_mask_list.append(human_mask)
    
    def _genGroup(self):
        i = 0
        while i < len(self.image_list):
            if len(self.image_list) - i == 1:
                break
            elif len(self.image_list) - i <= 3:
                self.image_group_list.append(self.image_list[i:])
                self.label_group_list.append(self.label_list[i:])
                self.human_mask_group_list.append(self.human_mask_list[i:])
                self.bbox_group_list.append(self.bbox_list[i:])
                self.human_bbox_group_list.append(self.human_bbox_list[i:])
                break

            num = 2 if random.random() < 0.5 else 3

            self.image_group_list.append(self.image_list[i:i+num])
            self.label_group_list.append(self.label_list[i:i+num])
            self.human_mask_group_list.append(self.human_mask_list[i:i+num])
            self.bbox_group_list.append(self.bbox_list[i:i+num])
            self.human_bbox_group_list.append(self.human_bbox_list[i:i+num])

            i += num

    def _processWithThreePerson(self, imgs, labels, human_masks, human_bboxes, core_image, core_label, core_human_box):
        for i in range(1, len(imgs)):
            alpha_img = labels[i]
            rgb_img = imgs[i]
            human_img = human_masks[i]

            xmin, ymin, xmax, ymax = human_bboxes[i]
            alpha_img = alpha_img[ymin:min(ymax, alpha_img.shape[1]), xmin:min(xmax, alpha_img.shape[0])]
            rgb_img = rgb_img[ymin:min(ymax, rgb_img.shape[1]), xmin:min(xmax, rgb_img.shape[0])]
            human_img = human_img[ymin:min(ymax, human_img.shape[1]), xmin:min(xmax, human_img.shape[0])]

            h, w, _ = rgb_img.shape

            alpha_img = alpha_img.astype('uint8')
            if i == 1:
                x = core_human_box[2] + 1
                y = core_human_box[1] + 1
            elif core_human_box[0] < w:
                return None, None
            else:
                x = random.randint(1, core_human_box[0] // 2)
                y = random.randint(1, h // 2)

            dst_mask = core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
            dst_img = core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
                
            if dst_mask.shape[0] == 0 or dst_mask.shape[1] == 0:
                return None, None

            rgb_img = cv2.resize(rgb_img, (dst_img.shape[1], dst_img.shape[0]))
            alpha_img = cv2.resize(alpha_img, (dst_img.shape[1], dst_img.shape[0]))
            human_img = cv2.resize(human_img, (dst_img.shape[1], dst_img.shape[0])) 

            final_mask = np.where(alpha_img != 0, alpha_img, dst_mask)
            final_img = np.where(alpha_img != 0, rgb_img, dst_img)
            final_img = np.where(human_img != 0, rgb_img, final_img)
         
            core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_img
            core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_mask

        return core_image, core_label

    def _processWithTwoPerson(self, imgs, labels, human_masks, human_bboxes, core_image, core_label, core_human_box, right_flag):
        alpha_img = labels[1]
        rgb_img = imgs[1]
        human_img = human_masks[1]

        xmin, ymin, xmax, ymax = human_bboxes[1]
        alpha_img = alpha_img[ymin:min(ymax, alpha_img.shape[1]), xmin:min(xmax, alpha_img.shape[0])]
        rgb_img = rgb_img[ymin:min(ymax, rgb_img.shape[1]), xmin:min(xmax, rgb_img.shape[0])]
        human_img = human_img[ymin:min(ymax, human_img.shape[1]), xmin:min(xmax, human_img.shape[0])]

        h, w, _ = rgb_img.shape

        alpha_img = alpha_img.astype('uint8')

        if core_human_box[0] < w and not right_flag:
            return None, None

        x = core_human_box[2] + 1 if right_flag else random.randint(1, core_human_box[0] // 2)
        y = core_human_box[1] + 1 if right_flag else random.randint(1, h // 2)

        dst_mask = core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
        dst_img = core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1])]
                
        if dst_mask.shape[0] == 0 or dst_mask.shape[1] == 0:
            return None, None

        rgb_img = cv2.resize(rgb_img, (dst_img.shape[1], dst_img.shape[0]))
        alpha_img = cv2.resize(alpha_img, (dst_img.shape[1], dst_img.shape[0]))
        human_img = cv2.resize(human_img, (dst_img.shape[1], dst_img.shape[0]))

                

        final_mask = np.where(alpha_img != 0, alpha_img, dst_mask)
        final_img = np.where(alpha_img != 0, rgb_img, dst_img)
        final_img = np.where(human_img != 0, rgb_img, final_img)
         
        core_image[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_img
        core_label[y:min(y+h, core_image.shape[0]), x:min(x+w, core_image.shape[1]), :] = final_mask

        return core_image, core_label


    def process(self):
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
                core_image, core_label = self._processWithThreePerson(imgs, labels, human_masks, human_bboxes, core_image, core_label, core_human_box)
            else:
                core_image, core_label = self._processWithTwoPerson(imgs, labels, human_masks, human_bboxes, core_image, core_label, core_human_box, right_flag)
            
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
        img_list = []
        files = os.listdir(self.config.input_image_dir)
        for i, f in enumerate(files):
            if os.path.isdir(os.path.join(self.config.input_image_dir, f)):
                LOG.logE('{} is a dir, {} contain a sub dir, you must get rid of it.'.format(os.path.join(self.config.input_image_dir, f), self.config.input_image_dir), exit=True)
            if os.path.isdir(os.path.join(self.config.input_label_dir, f.replace('jpg', 'png'))):
                LOG.logE('{} is a dir, {} contain a sub dir, you must get rid of it.'.format(os.path.join(self.config.input_label_dir, f.replace('jpg', 'png')), self.config.input_label_dir), exit=True)
            if os.path.isdir(os.path.join(self.config.input_human_mask_dir, f.replace('jpg', 'png'))):
                LOG.logE('{} is a dir, {} contain a sub dir, you must get rid of it.'.format(os.path.join(self.config.input_human_mask_dir, f.replace('jpg', 'png')), self.config.input_human_mask_dir), exit=True)
            self.image_path_list.append(os.path.join(self.config.input_image_dir, f))
            self.label_path_list.append(os.path.join(self.config.input_label_dir, f.replace('jpg', 'png')))
            self.human_mask_path_list.append(os.path.join(self.config.input_human_mask_dir, f.replace('jpg', 'png')))
        LOG.logI('Length of image_path_list is {} ...'.format(len(self.image_path_list)))


    def __call__(self):

        LOG.logI('Filter images begin...')
        self._filterImage()
        LOG.logI('Filter images finish...')

        LOG.logI('Generate group begin...')
        self._genGroup()
        LOG.logI('Generate group finish...')

        LOG.logI('Process begin...')
        self.process()
        LOG.logI('Process finish...')

        # self.saveImage()

if __name__ == "__main__":
    from config import config as deepvac_config
    synthesis = Synthesis(deepvac_config.synthesis)
    synthesis()