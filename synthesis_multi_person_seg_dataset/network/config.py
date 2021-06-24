"""
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
##############################################################################
# Config
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import torch


class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


__C = AttrDict()
cfg = __C

__C.OPTIONS = AttrDict()
__C.OPTIONS.TEST_MODE = False
__C.OPTIONS.TORCH_VERSION = None

__C.DATASET = AttrDict()
__C.DATASET.NUM_CLASSES = 2


__C.MODEL = AttrDict()
__C.MODEL.MSCALE = False
__C.MODEL.N_SCALES = None
__C.MODEL.OCR_ASPP = False
__C.MODEL.ASPP_BOT_CH = 256
__C.MODEL.BN = 'regularnorm'
__C.MODEL.SEGATTN_BOT_CH = 256
__C.MODEL.ALIGN_CORNERS = False
__C.MODEL.MSCALE_LO_SCALE = 0.5
__C.MODEL.MSCALE_DROPOUT = False
__C.MODEL.MSCALE_OLDARCH = False
__C.MODEL.MSCALE_INNER_3x3 = True
__C.MODEL.BNFUNC = torch.nn.BatchNorm2d

__C.MODEL.OCR = AttrDict()
__C.MODEL.OCR.MID_CHANNELS = 512
__C.MODEL.OCR.KEY_CHANNELS = 256

__C.MODEL.OCR_EXTRA = AttrDict()
__C.MODEL.OCR_EXTRA.FINAL_CONV_KERNEL = 1
__C.MODEL.OCR_EXTRA.STAGE1 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE1.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE1.NUM_RANCHES = 1
__C.MODEL.OCR_EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
__C.MODEL.OCR_EXTRA.STAGE1.NUM_BLOCKS = [4]
__C.MODEL.OCR_EXTRA.STAGE1.NUM_CHANNELS = [64]
__C.MODEL.OCR_EXTRA.STAGE1.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE2 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE2.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BRANCHES = 2
__C.MODEL.OCR_EXTRA.STAGE2.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
__C.MODEL.OCR_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
__C.MODEL.OCR_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE3 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE3.NUM_MODULES = 4
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BRANCHES = 3
__C.MODEL.OCR_EXTRA.STAGE3.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
__C.MODEL.OCR_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE4 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE4.NUM_MODULES = 3
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BRANCHES = 4
__C.MODEL.OCR_EXTRA.STAGE4.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
__C.MODEL.OCR_EXTRA.STAGE4.FUSE_METHOD = 'SUM'
