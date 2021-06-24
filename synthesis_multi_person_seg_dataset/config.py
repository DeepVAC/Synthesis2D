import torch
from torchvision import transforms as trans

from deepvac import AttrDict, new
from network import HRNet

config = new('PortraitSegTest')

config.core.PortraitSegTest.disable_git = True
config.core.PortraitSegTest.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
config.core.PortraitSegTest.net = HRNet(num_classes=num_classes)
config.core.PortraitSegTest.portrait_mask_output_dir = '<your-portrait-mask-dir>'
config.core.PortraitSegTest.model_path = 'weights/portrait.pth'
config.core.PortraitSegTest.test_loader = ''

config.core.PortraitSegTest.compose = trans.Compose([
    trans.Resize([448, 448]),
    trans.ToTensor(),
    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

config.core.Synthesis = config.core.PortraitSegTest.clone()
config.core.Synthesis.gen_portrait_mask = True
config.core.Synthesis.is_clothes_task = True # True: clothes , False: human

config.core.Synthesis.input_image_dir = '<your-input-image-dir>'
config.core.Synthesis.input_label_dir = '<your-input-label-dir>'

config.core.Synthesis.output_image_dir = '<your-output-image-dir>'
config.core.Synthesis.output_label_dir = '<your-output-label-dir>'