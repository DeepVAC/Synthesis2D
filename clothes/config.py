from deepvac import config, AttrDict

config.synthesis = AttrDict()

config.synthesis.input_image_dir = '<your-input-image-dir>'
config.synthesis.input_label_dir = '<your-input-label-dir>'
config.synthesis.input_human_mask_dir = '<your-input-human-mask-dir>'

config.synthesis.output_image_dir = '<your-output-image-dir>'
config.synthesis.output_label_dir = '<your-output-label-dir>'