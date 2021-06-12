import sys
import math
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from deepvac.backbones import Conv2dBNPReLU, Conv2dBN, BNPReLU, Conv2dDilatedBN, initWeightsKaiming

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([nn.Conv2d(features, features, 3, 1, 1, bias=False, groups=features) for size in sizes])
        self.project = Conv2dBNPReLU(features * (len(sizes) + 1), out_features, 1, 1)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(torch.cat(out, dim=1))

class EESP(nn.Module):
    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, shortcut=True):
        super(EESP, self).__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = Conv2dBNPReLU(nIn, n, 1, stride=1, groups=k)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = [int(3+2*i) if (3+2*i) <= r_lim else 3 for i in range(k)]
        self.k_sizes.sort()

        
        self.spp_dw = nn.ModuleList([nn.Conv2d(n, n, 3, stride, padding=map_receptive_ksize[i], dilation=map_receptive_ksize[i], groups=n, bias=False) for i in self.k_sizes])
        self.conv_1x1_exp = Conv2dBN(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BNPReLU(nOut)
        self.module_act = nn.PReLU(nOut)

        self.shortcut = shortcut

    def forward(self, input):
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)

        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))

        if self.shortcut:
            expanded = expanded + input

        return self.module_act(expanded)


class EESPDownSampler(EESP):
    def forward(self, input):
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)

        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        return expanded


class DownSampler(nn.Module):
    def __init__(self, nin, nout, k=4, r_lim=9, down_times=2):
        super(DownSampler, self).__init__()
        self.down_times = down_times
        nout_new = nout - nin
        self.eesp = EESPDownSampler(nin, nout_new, stride=2, k=k, r_lim=r_lim)
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.inp_reinf = nn.Sequential(Conv2dBNPReLU(3, 3, 3, 1), Conv2dBN(3, nout, 1, 1))
        self.act =  nn.PReLU(nout)

    def forward(self, input, input2=None):
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)

        for i in range(self.down_times):
            input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
        output = output + self.inp_reinf(input2)

        return self.act(output)

class EESPNet(nn.Module):
    def __init__(self):
        super(EESPNet, self).__init__()
        r_lim = [13, 11, 9, 7]
        config = [32, 128, 256, 512]

        channels = 3
        self.level1 = Conv2dBNPReLU(channels, config[0], 3, 2)
        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=r_lim[0], down_times=2)

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=r_lim[1], down_times=3)
        self.level3 = nn.Sequential(*[EESP(config[2], config[2], stride=1, k=4, r_lim=r_lim[2]) for i in range(3)])

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=r_lim[2], down_times=4)
        self.level4 = nn.Sequential(*[EESP(config[3], config[3], stride=1, k=4, r_lim=r_lim[3]) for i in range(7)])

        initWeightsKaiming(self)

    def forward(self, input):
        out_l1 = self.level1(input)

        out_l2 = self.level2_0(out_l1, input)

        out_l3_0 = self.level3_0(out_l2, input)
        out_l3 = self.level3(out_l3_0)

        out_l4_0 = self.level4_0(out_l3, input)
        out_l4 = self.level4(out_l4_0)

        return out_l1, out_l2, out_l3, out_l4

class EESPNet_Seg(nn.Module):
    def __init__(self, classes=20):
        super(EESPNet_Seg, self).__init__()
        self.net = EESPNet()

        self.proj_L4_C = Conv2dBNPReLU(self.net.level4[-1].module_act.num_parameters, self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2*self.net.level3[-1].module_act.num_parameters
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize //2, stride=1, k=4, r_lim=7, shortcut=False), PSPModule(pspSize // 2, pspSize //2))
        self.project_l3 = nn.Sequential(nn.Dropout2d(0.2), nn.Conv2d(pspSize // 2, classes, 1, 1, groups=1, bias=False))
        self.act_l3 = BNPReLU(classes)
        self.project_l2 = Conv2dBNPReLU(self.net.level2_0.act.num_parameters + classes, classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(0.2), nn.Conv2d(self.net.level1[2].num_parameters + classes, classes, 1, 1, groups=1, bias=False))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        return F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True), \
                F.interpolate(out_up_l2, scale_factor=2, mode='bilinear', align_corners=True), \
                proj_merge_l3_bef_act

if __name__ == '__main__':
    input = torch.Tensor(1, 3, 512, 1024).cuda()
    net = EESPNet_Seg(classes=20).cuda()
    out_x_8 = net(input)
    print(out_x_8[0].shape, out_x_8[1].shape)
