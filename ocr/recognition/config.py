from deepvac import AttrDict, new
config = new()
config.synthesis = AttrDict()

#synthesis
config.synthesis.total_num = 10
config.synthesis.txt_file = 'your lexicon txt'
config.synthesis.images_dir = 'your image path'
config.synthesis.video_file = 'your video path'
config.synthesis.sample_rate = 1
config.synthesis.fonts_dir = 'your font ttf path'
config.synthesis.chars = 'your char set path'
config.synthesis.chars_disturb_ratio = 0.1
config.synthesis.border_ratio = 0.3
config.synthesis.vertical_ratio = 0
config.synthesis.random_space_ratio = 0
config.synthesis.random_space_min = -0.1
config.synthesis.random_space_max = 0.1
config.synthesis.min_font = 8
config.synthesis.max_font = 60
