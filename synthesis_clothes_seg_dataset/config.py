import os
from deepvac import config, AttrDict
config.core.device = "cuda"
config.num_classes = 2
# config.skin_model = "weights/skin.pth"
config.input_size = (448, 448)
config.portrait_model = "weights/portrait.pth"
#whether to use portrait seg model to generate portriat mask and help to generate final dst mask.
config.use_portrait_mask = True
config.multiple = 2
#sys.argv[1] can override
config.original_image_label_dir = "/gemfield/hostpv2/deepvac_up_bottom_hat"
config.bg_dir = "/gemfield/hostpv2/standard_bg"
config.portrait_mask_output_dir = 'mask'
#sys.argv[2] can override
config.synthesis_output_dir = "/gemfield/hostpv2/deepvac_up_bottom_hat/synthesis"