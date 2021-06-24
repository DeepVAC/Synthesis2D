import torch 
from deepvac import AttrDict, new
from modules.model import RetinaFaceMobileNet, RetinaFaceResNet

from aug import SynthesisHatComposer

config = new('RetinaTest')

config.core.RetinaTest.disable_git = True
config.core.RetinaTest.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config.core.RetinaTest.model_path = "/ your face det model path /"
config.core.RetinaTest.confidence_threshold = 0.02
config.core.RetinaTest.nms_threshold = 0.4
config.core.RetinaTest.top_k = 5000
config.core.RetinaTest.keep_top_k = 1
config.core.RetinaTest.max_edge = 2000
config.core.RetinaTest.rgb_means = (104, 117, 123)
config.core.RetinaTest.net = RetinaFaceResNet()
config.core.RetinaTest.test_loader = ''

config.core.Synthesis2D = config.core.RetinaTest.clone()
config.core.Synthesis2D.input_image_dir = '/ your input image path /'

config.core.Synthesis2D.input_hat_mask_dir = '/ your hat mask path /'
config.core.Synthesis2D.input_hat_image_dir = '/ your hat(chartlet) image path /'

config.core.Synthesis2D.output_image_dir = '/ your output image path /'
config.core.Synthesis2D.output_anno_dir = '/ your output annotation path /'

config.core.Synthesis2D.compose = SynthesisHatComposer(config)

config.perspect_image_dir = '/ your perspect image dir /'
config.perspect_mask_dir = '/ your perspect mask dir /'
config.perspect_num = 5

config.flip_image_dir = '/ your flip image dir /'
config.flip_mask_dir = '/ your flip mask dir /'

config.fusion_hat_mask_dir = '/ your fusion hat mask dir /'
config.fusion_clothes_mask_dir = '/ your fusion clothes mask dir /'
config.fusion_new_mask_dir = '/ your fusion new(fusion) mask dir /'

config.generate_input_hat_image_dir_from_cocoannotator = '/ your input hat image dir /'
config.generate_input_hat_mask_dir_from_cocoannotator = '/ your input hat mask dir /'
config.generate_output_hat_image_dir = '/ your output hat image dir /'
config.generate_output_hat_mask_dir = '/ your output hat mask dir /'
