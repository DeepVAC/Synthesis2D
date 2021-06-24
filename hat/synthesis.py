import os
import time
import random
import math
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from deepvac import Deepvac, LOG
from deepvac.aug import PerspectAug
from deepvac.utils.face_utils import py_cpu_nms, decode, decode_landm, PriorBox

class RetinaTest(Deepvac):
    def __init__(self, deepvac_config):
        super(RetinaTest, self).__init__(deepvac_config)
        self.priorbox_cfgs = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'clip': False
        }
        self.variance = [0.1, 0.2]

    def auditConfig(self):
        pass

    def _pre_process(self, img_raw):
        h, w, c = img_raw.shape
        max_edge = max(h,w)
        if(max_edge > self.config.max_edge):
            img_raw = cv2.resize(img_raw,(int(w * self.config.max_edge / max_edge), int(h * self.config.max_edge / max_edge)))

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        img -= self.config.rgb_means
        img = img.transpose(2, 0, 1)
        self.input_tensor = torch.from_numpy(img).unsqueeze(0)
        self.input_tensor = self.input_tensor.to(self.config.device)
        self.img_raw = img_raw

    def _post_process(self, preds):
        loc, cls, landms = preds
        conf = F.softmax(cls, dim=-1)

        priorbox = PriorBox(self.priorbox_cfgs, image_size=(self.img_raw.shape[0], self.img_raw.shape[1]))
        priors = priorbox.forward()
        priors = priors.to(self.config.device)
        prior_data = priors.data
        resize = 1
        scale = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale = scale.to(self.config.device)
        boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        scale1 = torch.Tensor([self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2],
                        self.input_tensor.shape[3], self.input_tensor.shape[2], self.input_tensor.shape[3], self.input_tensor.shape[2],
                        self.input_tensor.shape[3], self.input_tensor.shape[2]])
        scale1 = scale1.to(self.config.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.config.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.config.top_k]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.config.keep_top_k, :]
        landms = landms[:self.config.keep_top_k, :]
        if len(dets)==0:
            return [], []

        bbox = dets[0]
        landmark = landms[0]
        
        return bbox, landmark

    def __call__(self, image):
        self._pre_process(image)

        tic = time.time()
        preds = self.config.net(self.input_tensor)
        end = time.time() - tic
        LOG.logI('net forward time: {:.4f}'.format(time.time() - tic))

        return self._post_process(preds)


class Synthesis2D(Deepvac):
    def __init__(self, deepvac_config):
        super(Synthesis2D, self).__init__(deepvac_config)
        self.retina_det = RetinaTest(config)

    def auditConfig(self):
        pass

    def _synthesisData(self, img_raw, rgb_hat, a):
        h, w, _ = rgb_hat.shape

        factor = random.uniform(2.0, 2.2)
        dh = random.randint(-5, 2)
        dw = random.randint(-3, 3)

        rgb_hat = self.config.compose(rgb_hat)

        bbox, landmark = self.retina_det(img_raw)
        if len(bbox) == 0:
            LOG.logI('error noface detect!')
            return None, None
    
        point1 = landmark[0:2]
        point2 = landmark[0:4]
    
        eyes_center = ((point1[0]+point2[0])//2,(point1[1]+point2[1])//2)
        x, y, b_w, b_h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]

        resized_hat_h = int(round(rgb_hat.shape[0]*b_w/rgb_hat.shape[1]*factor))
        resized_hat_w = int(round(rgb_hat.shape[1]*b_w/rgb_hat.shape[1]*factor))
  
        if resized_hat_w <= 0 or resized_hat_h <= 0:
            LOG.logI('error resized_hat_w')
            return None, None

        angle = math.atan2(point1[1]-point2[1], point2[0]-point1[0])
        angle = angle * 180 / 3.14

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 0, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rgb_hat = cv2.warpAffine(rgb_hat, M, (nW, nH), borderValue=(255,255,255))
        a = cv2.warpAffine(a, M, (nW, nH), borderValue=(0, 0, 0))
        h, w, _ = rgb_hat.shape

        a = a.astype('uint8')
        resized_hat = cv2.resize(rgb_hat,(int(resized_hat_w),int(resized_hat_h)))
        mask = cv2.resize(a,(int(0.7*resized_hat_w),int(0.7*resized_hat_h)))
    
        mask = cv2.copyMakeBorder(mask,int(0.3*resized_hat_h/2), resized_hat_h-int(0.7*resized_hat_h)-int(0.3*resized_hat_h/2),
            int(0.3*resized_hat_w/2),resized_hat_w-int(0.7*resized_hat_w)-int(0.3*resized_hat_w/2), cv2.BORDER_CONSTANT,value=[0,0,0])
    
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        retval, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_inv = mask

        mask_inv = np.where(mask_inv==0, 255, 0)
        mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))

        y_s = int(eyes_center[1]+dh-resized_hat_h) 
        y_s = y_s if y_s >0 else 0
        x_s = int(eyes_center[0]-resized_hat_w//3+dw) 
        x_s = x_s if x_s >0 else 0
        x_end    = eyes_center[0]+resized_hat_w//3*2+dw
        bg_roi = img_raw[y_s:int(eyes_center[1]+dh), x_s:int(x_end)]

        bg_roi = bg_roi.astype(float)
        alpha = mask_inv.astype(float)/255

        alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
        bg = cv2.multiply(alpha, bg_roi)

        bg = bg.astype('uint8')

        hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)

        hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))
        add_hat = cv2.add(bg,hat)
        mask_label = img_raw.copy()
        img_raw[y_s:int(eyes_center[1]+dh),x_s:int(x_end)] = add_hat
        mask_label[:,:,0] = 0
        mask_label[:,:,1] = 0
        mask_label[:,:,2] = 0
        bg_add = cv2.bitwise_not(bg)
        alpha = np.where(alpha==1, 0, 1)
        mask_label[y_s:int(eyes_center[1]+dh),x_s:int(x_end)] = alpha
        ah, aw, _ = add_hat.shape
        
        return img_raw, mask_label

    def doTest(self):
        input_image_dir = self.config.input_image_dir
        input_hat_image_dir = self.config.input_hat_image_dir
        input_hat_mask_dir = self.config.input_hat_mask_dir
        output_image_dir = self.config.output_image_dir
        output_anno_dir = self.config.output_anno_dir

        if not os.path.exists(self.config.output_image_dir):
            os.makedirs(self.config.output_image_dir)
        if not os.path.exists(self.config.output_anno_dir):
            os.makedirs(self.config.output_anno_dir)
        
        imgs = os.listdir(input_image_dir)
        hats = os.listdir(input_hat_image_dir)
    
        for i, img in enumerate(imgs):
            LOG.logI('num: {} ---- path: {}'.format(i, os.path.join(input_image_dir, img)))
        
            img_raw = cv2.imread('{}/{}'.format(input_image_dir, img))
        
            name = hats[random.randint(0, len(hats)-1)]

            rgb_hat = cv2.imread('{}/{}'.format(input_hat_image_dir, name))
            a = cv2.imread('{}/{}.png'.format(input_hat_mask_dir, os.path.splitext(name)[0]))
        
            new_img, mask_label = self._synthesisData(img_raw, rgb_hat, a)
        
            if new_img is None:
                continue
        
            cv2.imwrite("{}/{}".format(output_image_dir, img),new_img)
            cv2.imwrite("{}/{}.png".format(output_anno_dir, os.path.splitext(img)[0]),mask_label)
        self.config.sample = torch.rand((1, 3, 112, 112))

def generate(deepvac_config):
    imgs = os.listdir(deepvac_config.generate_input_hat_mask_dir_from_cocoannotator)

    for img in imgs:
        alpha = cv2.imread(os.path.join(deepvac_config.generate_input_hat_mask_dir_from_cocoannotator, img))
        ori_img = cv2.imread('{}/{}.jpg'.format(deepvac_config.generate_input_hat_dir_image_from_cocoannotator, os.path.splitext(img)[0]))        
        alpha[alpha!=1] = 0
        ori_img = cv2.multiply(ori_img, alpha)
        ori_img[ori_img==0] = 255
        alpha[alpha==1] = 255

        coor = np.nonzero(alpha)
        ymin = coor[0][0]
        ymax = coor[0][-1]
        coor[1].sort()
        xmin = coor[1][0]
        xmax = coor[1][-1]
        alpha = alpha[ymin:ymax, xmin:xmax]        
        ori_img = ori_img[ymin:ymax, xmin:xmax]

        cv2.imwrite(os.path.join(deepvac_config.generate_output_hat_mask_dir, img), alpha)
        cv2.imwrite('{}/{}.jpg'.format(deepvac_config.generate_output_hat_image_dir, os.path.splitext(img)[0]), ori_img)


def perspect(deepvac_config):
    names = os.listdir(deepvac_config.perspect_image_dir)

    perspect_aug = PerspectAug(deepvac_config)
    
    for name in names:
        for i in range(deepvac_config.perspect_num):
            img = cv2.imread('{}/{}.jpg'.format(deepvac_config.perspect_image_dir, os.path.splitext(name)[0]))
            mask = cv2.imread('{}/{}.png'.format(deepvac_config.perspect_mask_dir, os.path.splitext(name)[0]))
            
            perspect_aug.config.borderValue = (255, 255, 255)
            perspect_img = perspect_aug(img)

            perspect_aug.config.borderValue = (0, 0, 0)
            perspect_mask = perspect_aug(mask)

            cv2.imwrite('{}/{}_{}.jpg'.format(deepvac_config.perspect_image_dir, os.path.splitext(name)[0], i+1), perspect_img)
            cv2.imwrite('{}/{}_{}.png'.format(deepvac_config.perspect_mask_dir, os.path.splitext(name)[0], i+1), perspect_mask)

def flip(deepvac_config):
    names = os.listdir(deepvac_config.flip_image_dir)
    for name in names:
        img = cv2.imread('{}/{}.jpg'.format(deepvac_config.flip_image_dir, os.path.splitext(name)[0]))
        mask = cv2.imread('{}/{}.png'.format(deepvac_config.flip_mask_dir, os.path.splitext(name)[0]))
        img_flip = cv2.flip(img, 1)
        mask_flip = cv2.flip(mask, 1)

        cv2.imwrite('{}/{}_flip.jpg'.format(deepvac_config.flip_image_dir, os.path.splitext(name)[0]), img_flip)
        cv2.imwrite('{}/{}_flip.png'.format(deepvac_config.flip_mask_dir, os.path.splitext(name)[0]), mask_flip)

def checkMask(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]

def fusion(deepvac_config):
    if not os.path.exists(deepvac_config.fusion_new_mask_dir):
        os.makedirs(deepvac_config.fusion_new_mask_dir)

    hats = os.listdir(deepvac_config.fusion_hat_mask_dir)

    for fn in hats:
        fp = os.path.join(deepvac_config.fusion_hat_mask_dir, fn)
        mask_hat = cv2.imread(fp)
        checkMask(mask_hat)

        fp = os.path.join(deepvac_config.fusion_clothes_mask_dir, fn)
        mask_clothes = cv2.imread(fp)
        checkMask(mask_clothes)

        mask = np.clip(mask_hat + mask_clothes, 0, 3)
        cv2.imwrite(os.path.join(deepvac_config.fusion_new_mask_dir, fn), mask)

if __name__ == '__main__':
    from config import config as deepvac_config
    import sys
    if len(sys.argv) != 2:
        LOG.logE("Usage: python synthesis.py <synthesis|perspect|fusion|flip|generate>", exit=True)

    op = sys.argv[1]

    if op not in ('synthesis', 'perspect', 'fusion', 'flip', 'generate'):
        LOG.logE("Usage: python synthesis.py <synthesis|perspect|fusion|flip|generate>", exit=True)

    if op == 'synthesis':
        synthesis = Synthesis2D(deepvac_config)
        synthesis()
    
    if op == 'perspect':
        perspect(deepvac_config)

    if op == 'fusion':
        fusion(deepvac_config)
    
    if op == 'flip':
        flip(deepvac_config)

    if op == 'generate':
        generate(deepvac_config)
