from deepvac import config, AttrDict

config.core.device = "cuda"
config.num_classes = 2
config.input_size = (448, 448)
config.portrait_model = "weights/portrait.pth"
config.use_portrait_mask = False
config.synthesis = AttrDict()

config.synthesis.is_clothes_task = True # True: clothes , False: human

config.synthesis.input_image_dir = '<your-input-image-dir>'
config.synthesis.input_label_dir = '<your-input-label-dir>'
config.synthesis.portrait_mask_output_dir = '<your-portrait-mask-dir>'

config.synthesis.output_image_dir = '<your-output-image-dir>'
config.synthesis.output_label_dir = '<your-output-label-dir>'