import torch
import torch.nn as nn
import torchvision.models._utils as _utils

from deepvac.syszux_modules import SSH, FPN
from deepvac.syszux_mobilenet import MobileNetV3Large
from deepvac.syszux_resnet import ResNet50
from deepvac.syszux_regnet import RegNetSmall

from deepvac.syszux_post_process import py_cpu_nms, decode, decode_landm, PriorBox
import torch.nn.functional as F
import numpy as np

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFaceMobileNetBackbone(MobileNetV3Large):
    def __init__(self):
        super(RetinaFaceMobileNetBackbone, self).__init__()
        self.return_layers = [5, 10, 15]

    def initFc(self):
        pass
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = []
        for i, fea in enumerate(self.features):
            x = fea(x)
            if i in self.return_layers:
                out.append(x)
        #x = self.conv(x)
        #x = self.pool(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return out

class RetinaFaceMobileNet(nn.Module):
    def __init__(self):
        super(RetinaFaceMobileNet, self).__init__()
        self.auditConfig()

        self.fpn = FPN(self.in_channels_list, self.out_channels)
        self.ssh1 = SSH(self.out_channels, self.out_channels)
        self.ssh2 = SSH(self.out_channels, self.out_channels)
        self.ssh3 = SSH(self.out_channels, self.out_channels)
        
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=self.out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=self.out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=self.out_channels)
    
    def auditConfig(self):
        self.in_channels_list = [40, 80, 160]
        self.out_channels = 64
        self.body = RetinaFaceMobileNetBackbone()

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead

    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        return (bbox_regressions, classifications, ldm_regressions)

class RetinaFaceResNetBackbone(ResNet50):
    def initFc(self):
        self.return_layers = [6, 12, 15]
    
    def forward(self, x):
        out = []
        index = 1
        x = self.conv1(x)
        x = self.maxpool(x)
        for i, lay in enumerate(self.layer):
            x = lay(x)
            if i in self.return_layers:
                out.append(x)
        x = self.avgpool(x)
        return out

class RetinaFaceResNet(RetinaFaceMobileNet):
    def __init__(self):
        super(RetinaFaceResNet, self).__init__()

    def auditConfig(self):
        self.in_channels_list = [512, 1024, 2048]
        self.out_channels = 256
        self.body = RetinaFaceResNetBackbone()

class RetinaFaceRegNetBackbone(RegNetSmall):
    def initFc(self):
        self.return_layers = [0, 1, 2]
        self.layers = [self.s1, self.s2, self.s3, self.s4]

    def forward(self, x):
        self.return_layers = [1, 2, 3]
        out = []
        index = 1
        x = self.stem(x)
        #x = self.maxpool(x)
        for i, cur in enumerate([self.s1, self.s2, self.s3, self.s4]):
            #for j, lay in enumerate(cur):
            x = cur(x)
            if i in self.return_layers:
                out.append(x)

        #x = self.head(x)
        return out

class RetinaFaceRegNet(RetinaFaceMobileNet):
    def __init__(self):
        super(RetinaFaceRegNet, self).__init__()
        #self.device = device
    def auditConfig(self):
        self.in_channels_list = [104, 208, 440]
        self.out_channels = 64
        self.body = RetinaFaceRegNetBackbone()

