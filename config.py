import torch 
from deepvac import config, AttrDict

config.disable_git = True
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config.model_path = "/ your face det model path /"
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.confidence_threshold = 0.02
config.nms_threshold = 0.4
config.top_k = 5000
config.keep_top_k = 1
config.max_edge = 2000
config.rgb_means = (104, 117, 123)

config.input_img_dir = '/ your input image path /'
config.input_mask_dir = '/ your mask image path /'

config.hat_dir = '/ your hat(chartlet) path /'

config.to_image_dir = '/ your result image path /'
config.to_anno_dir = '/ your result annotation path /'
