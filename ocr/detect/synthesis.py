from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import os
import random
import time
import math
from deepvac import LOG
try:
    from fontTools.ttLib import TTFont
except:
    LOG.logE('no fonttools, pip install fonttools please', exit=True)

class SynthesisBase(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.auditConfig()

    def auditConfig(self):
        self.total_num = self.conf.total_num
        self.output_dir = self.conf.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __call__():
        pass

    def dumpImgToPath(self, file_name, img):
        output_file_name = os.path.join(self.output_dir, file_name)
        try:
            cv2.imwrite(output_file_name, img)
        except:
            target_dir = os.path.dirname(os.path.abspath(output_file_name))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            cv2.imwrite(output_file_name, img)

class SynthesisText(SynthesisBase):
    def __init__(self, deepvac_config):
        super(SynthesisText, self).__init__(deepvac_config)
        self.i_just_want_font = None

    def getFontTables(self, font_path):
        return TTFont(font_path)

    def hasGlyph(self, font, glyph):
        for table in font['cmap'].tables:
            if ord(glyph) in table.cmap.keys():
                return True
        return False

    def auditConfig(self):
        super(SynthesisText, self).auditConfig()
        self.lex = []
        self.pil_img = None
        self.draw = None
        self.txt_file = self.conf.txt_file
        assert os.path.isfile(self.txt_file), "txt file {} not exist.".format(self.txt_file)

        with open(self.txt_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                self.lex.append(line)
        random.shuffle(self.lex)
        self.lex_len = len(self.lex)

        self.fonts_dir = self.conf.fonts_dir
        if not os.path.exists(self.fonts_dir):
            raise Exception("Dir {} not found!".format(self.fonts_dir))
        self.font_file_list = os.listdir(self.fonts_dir)
        self.font_num = len(self.font_file_list)
        if self.font_num == 0:
            raise Exception("No font was found in {}!".format(self.fonts_dir))
        self.current_font_size = 50
        self.max_font = 60 if self.conf.max_font is None else self.conf.max_font 
        self.min_font = 15 if self.conf.min_font is None else self.conf.min_font 
        self.crop_scale = 8
        self.scene_hw = (1080,1920)
        self.gb18030_font_file_list = []

        for i, font_file in enumerate(self.font_file_list):
            LOG.logI("found font: {}:{}".format(i, font_file))
            if font_file.startswith('gb18030'):
                LOG.logI("And this is a gb18030 font!")
                self.gb18030_font_file_list.append(font_file)

        self.runtime_fonts = dict()
        self.runtime_gb18030_fonts = dict()
        for font_size in range (self.min_font, self.max_font + 1):
            self.runtime_fonts[font_size] = []
            for font_file in self.font_file_list:
                font = ImageFont.truetype(os.path.join(self.fonts_dir,font_file), font_size, encoding='utf-8')
                self.runtime_fonts[font_size].append(font)

        for font_size in range (self.min_font, self.max_font + 1):
            self.runtime_gb18030_fonts[font_size] = []
            for font_file in self.gb18030_font_file_list:
                font = ImageFont.truetype(os.path.join(self.fonts_dir,font_file), font_size, encoding='utf-8')
                self.runtime_gb18030_fonts[font_size].append(font)

        self.support_fonts4char = {}
        with open(self.conf.chars, 'r') as f:
            line = f.readlines()[0][:-1]
            for c in line:
                self.support_fonts4char[c]= []
            for idx, font_file in enumerate(self.font_file_list):
                font_table = self.getFontTables(os.path.join(self.fonts_dir,font_file))
                for c in line:
                    if self.hasGlyph(font_table, c):
                        self.support_fonts4char[c].append(idx)
                    elif font_file in self.gb18030_font_file_list:
                        LOG.logE('{} not supported in current font! Are you sure {} is a gb18030 font?'.format(c, font_file))

        self.fg_color = [(10,10,10),(200,10,10),(10,10,200),(200,200,10),(255,255,255)]
        self.fg_color_len = len(self.fg_color)

        self.distance = 100  # The min distance of fg_color and bg_color 

        self.s_width = 0
        self.s_height = 0

        self.min_text_num = 5
        self.max_text_num = 15
        self.dense_text_num = 50

        self.min_rotate_angle = -90
        self.max_rotate_angle = 90

        self.integral_thre_try_num = 10
        self.each_thre_try_num = 10
        self.box_overlap_try_num = 10
        self.integral_init_thre = 1.0
        self.integral_thre_step = 2.0

    def buildScene(self,i):
        raise Exception("Not implemented!")
    
    def buildTextWithScene(self, i):
        raise Exception("Not implemented!")

    def setCurrentFontSizeAndGetFont(self, i):
        s = self.lex[i%self.lex_len]
        self.current_font_size = np.random.randint(self.min_font,self.max_font+1)
        if self.i_just_want_font is not None:
            return s, self.runtime_fonts[self.current_font_size][self.i_just_want_font % self.font_num]
        
        font_idx = i%self.font_num
        font = self.runtime_fonts[self.current_font_size][i%self.font_num]
        for c in s:
            if font_idx in self.support_fonts4char[c]:
                continue
            font = self.runtime_gb18030_fonts[self.current_font_size][np.random.randint(0,len(self.runtime_gb18030_fonts[self.current_font_size]))]
            break

        return s, font

    def drawText(self, font_offset, font, fillcolor, s):
        # vertical font
        is_vertical = False
        if np.random.rand() < self.conf.vertical_ratio:
            font = ImageFont.TransposedFont(font, orientation=Image.ROTATE_90)
            is_vertical = True

        # border
        is_border = False
        if np.random.rand() < self.conf.border_ratio:
            shadowcolor = 'black' if fillcolor==(255,255,255) else 'white'
            is_border = True

        # random_space
        is_random_space = False
        char_space_width = 0
        if np.random.rand() < self.conf.random_space_ratio:
            is_random_space = True


        chars_size = []
        width = 0
        height = 0
        y_offset = 10 ** 5
        for c in s:
            size = font.getsize(c)
            chars_size.append(size)
            width += size[0]

            if size[1] > height:
                height = size[1]

            if is_vertical:
                c_offset = font.font.getoffset(c)
                if c_offset[0] < y_offset:
                    y_offset = c_offset[0]
            else:
                c_offset = font.getoffset(c)
                if c_offset[1] < y_offset:
                    y_offset = c_offset[1]

        c_x, c_y = font_offset
        c_x_ori = c_x
        c_y_ori = c_y
        c_y -= y_offset

        char_space_width = int(height * np.random.uniform(self.conf.random_space_min, self.conf.random_space_max)) if is_random_space else 0
        width += (char_space_width * (len(s) - 1))
        height -= y_offset

        if not is_vertical and not is_random_space:
            s = [s]

        for i, c in enumerate(s):
            if is_border:
                x = c_x
                y = c_y
                for j in [x-1,x+1,x]:
                    for k in [y-1,y+1,y]:
                        self.draw.text((j, k), c, font=font, fill=shadowcolor)
            self.draw.text((c_x, c_y), c, fillcolor, font=font)
            c_x += (chars_size[i][0] + char_space_width)

        self.s_width = width
        self.s_height = height
        self.font_size_list.append(self.s_height)
        self.font_length_list.append(self.s_width)

    def getRotateCoord(self, wh, angle):
        height = wh[1]
        width = wh[0]
        pt1 = [0, 0]
        pt2 = [width, 0]
        pt3 = [width, height]
        pt4 = [0, height]

        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        heightNew = int(
            width * math.fabs(math.sin(math.radians(angle))) + height * math.fabs(math.cos(math.radians(angle))))
        widthNew = int(height * math.fabs(math.sin(math.radians(angle))) + width * math.fabs(math.cos(math.radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2

        coord1 = np.maximum(np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]])).reshape(2), 0).astype(np.int)
        coord2 = np.maximum(np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]])).reshape(2), 0).astype(np.int)
        coord3 = np.maximum(np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]])).reshape(2), 0).astype(np.int)
        coord4 = np.maximum(np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]])).reshape(2), 0).astype(np.int)
        return  np.array([coord1, coord2, coord3, coord4])

    def buildFGImages(self):
        self.is_dense = False
        self.use_same_font = False
        self.text_num = np.random.randint(self.min_text_num,self.max_text_num+1)
        if np.random.rand() < self.conf.dense_ratio:
            self.is_dense = True
            self.text_num = self.dense_text_num 
        self.fg_images = []
        self.text_coordinates_in_fg = []
        for _ in range(self.text_num):
            img = Image.new("RGBA", (self.max_font*30, self.max_font+5), color=(0, 0, 0, 0))
            self.fg_images.append(img)
        self.angle = np.random.randint(self.min_rotate_angle, self.max_rotate_angle + 1)
        if np.random.rand() < self.conf.same_font_ratio:
            self.use_same_font = True
        self.font_size_list = []
        self.font_length_list = []

    def randomRotateImage(self, i):
        self.fg_images[i] = self.fg_images[i].crop((0, 0, self.s_width, self.s_height))
        if not self.is_dense:
            self.angle = np.random.randint(self.min_rotate_angle, self.max_rotate_angle + 1)
        wh = self.fg_images[i].size[:2]
        self.fg_images[i] = self.fg_images[i].rotate(self.angle, expand=True)
        self.text_coordinates_in_fg.append(self.getRotateCoord(wh, self.angle))

    def dumpTextImg(self,i):
        box_num = 0
        txt_name = os.path.join(self.output_dir, '{}_{}.txt'.format(self.dump_prefix,str(i).zfill(6)))
        fw = open(txt_name, 'w')
        for idx, fg_image in enumerate(self.fg_images):
            box = self.text_boxes[idx]
            if box is None:
                continue
            self.pil_img.paste(fg_image, (box[0], box[1]), fg_image.split()[3])

            coord = self.text_coordinates_in_fg[idx] + [box[0], box[1]]

            s = ""
            for pt in coord:
                x, y = pt
                s = s+str(x)+','+str(y)+','
            s += '0'
            fw.write(s+'\n')
            box_num += 1

        pasted_img = cv2.cvtColor(np.asarray(self.pil_img),cv2.COLOR_RGB2BGR)
        image_name = '{}_{}.jpg'.format(self.dump_prefix,str(i).zfill(6))
        self.dumpImgToPath(image_name, pasted_img)

        if box_num<=0:
            LOG.logW("Image {} paste no text.".format(i))

    def checkBoxIsOverlap(self, b_box):
        for a_box in self.text_boxes:
            if a_box == None:
                continue
            in_h = min(a_box[2], b_box[2]) - max(a_box[0], b_box[0])
            in_w = min(a_box[3], b_box[3]) - max(a_box[1], b_box[1])
            inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
            if inter > 0:
                return True
        return False

    def genCandidateBox(self, bg_hw, fg_hw):
        for _ in range(self.box_overlap_try_num):
            x1 = np.random.randint(0, bg_hw[1] - fg_hw[1])
            x2 = x1+fg_hw[1]
            y1 = np.random.randint(0, bg_hw[0] - fg_hw[0])
            y2 = y1+fg_hw[0]
            candidate_box = [x1, y1, x2, y2]
            if not self.checkBoxIsOverlap(candidate_box):
                return candidate_box
        return None

    def genTextBox(self, integral_thre, bg_hw, fg_hw):
        for try_num in range(self.each_thre_try_num):
            candidate_box = self.genCandidateBox(bg_hw, fg_hw)
            if candidate_box is not None:
                x1, y1, x2, y2 = candidate_box
                integral = self.image_integral[y2][x2] - self.image_integral[y2][x1] - self.image_integral[y1][x2] + self.image_integral[y1][x1]
                if integral<=integral_thre:
                    return candidate_box
        return None

    def genTextBoxes(self):
        if len(self.fg_images) <= 0:
            return
        bg_h, bg_w = self.scene_hw
        for fg_image in self.fg_images:
            fg_w, fg_h = fg_image.size[:2]
            for try_num in range(self.integral_thre_try_num):
                text_box = self.genTextBox(self.integral_init_thre+try_num*self.integral_thre_step, (bg_h, bg_w), (fg_h, fg_w))
                if text_box is not None or try_num==self.integral_thre_try_num - 1:
                    self.text_boxes.append(text_box)
                    break

    def genDenseTextBoxes(self):
        if len(self.fg_images) <= 0:
            return
        aligin = 100
        initial_box = [aligin,aligin,aligin+self.fg_images[0].size[0],aligin+self.fg_images[0].size[1]]
        self.text_boxes.append(initial_box)
        already_double_column = False
        sina = math.sin(math.pi*self.angle/180)
        cosa = math.cos(math.pi*self.angle/180)
        for idx, fg_image in enumerate(self.fg_images):
            if idx==0:
                continue
            fg_w, fg_h = fg_image.size[:2]
            if not self.text_boxes[idx-1]:
                self.text_boxes.append(None)
                continue
            prev_x, prev_y = self.text_boxes[idx-1][:2]

            prev_font_size = self.font_size_list[idx-1]
            cur_font_size = self.font_size_list[idx]
            if self.angle<=0:
                if abs(self.angle)<=45:
                    prev_y += prev_font_size*cosa + cur_font_size*(1/cosa-cosa)
                else:
                    prev_x += prev_font_size*abs(sina) + cur_font_size*(1/abs(sina)-abs(sina))
            else:
                prev_length = self.font_length_list[idx-1]
                cur_length = self.font_length_list[idx]
                if abs(self.angle)<=45:
                    prev_y += (prev_length-cur_length)*abs(sina)+prev_font_size/cosa
                else:
                    prev_x += (prev_length-cur_length)*cosa+prev_font_size/abs(sina)
            prev_x, prev_y = int(prev_x), int(prev_y)
            if prev_x<0 or prev_y<0 or prev_x+fg_image.size[0]>=self.frame.shape[1] or prev_y+fg_image.size[1]>=self.frame.shape[0]:
                if not already_double_column and abs(self.angle)<=45:
                    already_double_column = True
                    self.text_boxes.append([aligin+int(self.frame.shape[1]/2), aligin, aligin+int(self.frame.shape[1]/2)+fg_image.size[0], aligin+fg_image.size[1]])
                else:
                    self.text_boxes.append(None)
                continue
            self.text_boxes.append([prev_x, prev_y, prev_x+fg_image.size[0], prev_y+fg_image.size[1]])

    def buildPasteArea(self, i):
        self.text_boxes = []
        if self.is_dense:
            # paste dense text 
            self.genDenseTextBoxes()
        else:
            kp_image = cv2.Canny(self.frame, 50, 150) // 255
            sum_arr = np.zeros(self.scene_hw, np.float32)
            self.image_integral = cv2.integral(kp_image, sum_arr, cv2.CV_32FC1)
            self.genTextBoxes()

    def __call__(self):
        start_time = time.time()
        for i in range(self.total_num):
            # prepare for background image
            self.buildScene(i)
            # prepare for front transparent images (including define s, font, font_size, generate transparent images)
            self.buildTextWithScene(i)
            # prepare background image coordinate to paste transparent images
            self.buildPasteArea(i)
            # paste transparent images to background image and save results
            self.dumpTextImg(i)
            if i%5000==0:
                LOG.logI('{}/{}'.format(i,self.total_num))
        LOG.logI("Total time: {}".format(time.time() - start_time))

class SynthesisTextPure(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextPure, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()
    
    def auditConfig(self):
        super(SynthesisTextPure, self).auditConfig()
        self.bg_color = [(255,255,255),(10,10,200),(200,10,10),(10,10,200),(10,10,10)]
        self.bg_color_len = len(self.bg_color)
        self.font_offset = (0, 0)
        self.is_border = self.conf.is_border
        self.fw = open(os.path.join(self.conf.output_dir,'pure.txt'),'w')
        self.dump_prefix = 'pure'

    def buildScene(self, i):
        r_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][0]
        g_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][1]
        b_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][2]
        self.frame = cv2.merge((b_channel, g_channel, r_channel))
        self.pil_img = Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB))
        self.buildFGImages()

    def buildTextWithScene(self, i):
        for idx in range(self.text_num):
            if not self.use_same_font or idx==0:
                s, font = self.setCurrentFontSizeAndGetFont(np.random.randint(0, self.lex_len + 1))
                fillcolor = self.fg_color[i%self.fg_color_len]
            s, _ = self.setCurrentFontSizeAndGetFont(np.random.randint(0, self.lex_len + 1))
            self.draw = ImageDraw.Draw(self.fg_images[idx])
            self.drawText(self.font_offset,font,fillcolor,s)
            self.randomRotateImage(idx)

class SynthesisTextFromVideo(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextFromVideo, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()

    def auditConfig(self):
        super(SynthesisTextFromVideo, self).auditConfig()
        self.video_file = self.conf.video_file

        self.video_capture = cv2.VideoCapture(self.video_file)
        self.frames_num = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        assert self.frames_num > 10, "invalid video file {}".format(self.video_file)
        self.sample_rate = self.conf.sample_rate
        if self.frames_num/self.sample_rate<self.total_num:
            raise Exception("Total_num {} exceeds frame_nums({})/sample_rate({}), build exit!".format(self.total_num,int(self.frames_num),self.sample_rate))
        self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        assert self.frame_height > 4 * self.max_font, "video height must exceed {} pixels".format(4*self.max_font)
        self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.resize_ratio = None

        if self.frame_width < self.scene_hw[1]:
            self.resize_ratio = self.scene_hw[1] / self.frame_width
            self.frame_height = int(self.frame_height * self.resize_ratio)

        self.font_offset = (0, 0)
        self.is_border = self.conf.is_border
        self.dump_prefix = 'scene'
        self.fw = open(os.path.join(self.conf.output_dir,'video.txt'),'w')
        

    def buildScene(self, i):
        for _ in range(self.sample_rate):
            success,self.frame = self.video_capture.read()

            if not success:
                return
        self.frame = cv2.resize(self.frame, (self.scene_hw[1],self.scene_hw[0]))

        self.pil_img = Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)
        self.buildFGImages()

    def buildTextWithScene(self, i):
        for idx in range(self.text_num):
            if not self.use_same_font or idx==0:
                s, font = self.setCurrentFontSizeAndGetFont(np.random.randint(0, self.lex_len + 1))
                fillcolor = self.fg_color[np.random.randint(0, self.fg_color_len)]
            s, _ = self.setCurrentFontSizeAndGetFont(np.random.randint(0, self.lex_len + 1))
            self.draw = ImageDraw.Draw(self.fg_images[idx])
            self.drawText(self.font_offset,font,fillcolor,s)
            self.randomRotateImage(idx)

class SynthesisTextFromImage(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextFromImage, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()

    def auditConfig(self):
        super(SynthesisTextFromImage, self).auditConfig()
        self.images_dir = self.conf.images_dir
        if not os.path.exists(self.images_dir):
            raise Exception("Dir {}not found!".format(self.images_dir))
        self.images = os.listdir(self.images_dir)
        self.images_num = len(self.images)
        if self.images_num==0:
            raise Exception("No image was found in {}!".format(self.images))
        self.font_offset = (0, 0)
        self.is_border = self.conf.is_border
        self.dump_prefix = 'image'
        self.fw = open(os.path.join(self.conf.output_dir,'image.txt'),'w')

    def buildScene(self, i):
        image = cv2.imread(os.path.join(self.images_dir, self.images[i%self.images_num]))
        self.frame = cv2.resize(image,(self.scene_hw[1],self.scene_hw[0]))
        self.pil_img = Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)
        self.buildFGImages()

    def buildTextWithScene(self, i):
        for idx in range(self.text_num):
            if not self.use_same_font or idx==0:
                s, font = self.setCurrentFontSizeAndGetFont(np.random.randint(0, self.lex_len + 1))
                fillcolor = self.fg_color[np.random.randint(0, self.fg_color_len)]
            s, _ = self.setCurrentFontSizeAndGetFont(np.random.randint(0, self.lex_len + 1))
            self.draw = ImageDraw.Draw(self.fg_images[idx])
            self.drawText(self.font_offset,font,fillcolor,s)
            self.randomRotateImage(idx)

if __name__ == '__main__':
    from config import config as deepvac_config

    # from Image
    deepvac_config.synthesis.output_dir = 'image'
    gen = SynthesisTextFromImage(deepvac_config.synthesis)
    gen()

    # from video 
    deepvac_config.synthesis.output_dir = 'video'
    gen = SynthesisTextFromVideo(deepvac_config.synthesis)
    gen()

    # from pure
    deepvac_config.synthesis.output_dir = 'pure'
    gen = SynthesisTextPure(deepvac_config.synthesis)
    gen()
