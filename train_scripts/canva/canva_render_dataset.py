import json
import os.path as osp
import random
import string
import copy


from PIL import ImageDraw, ImageFont, Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from dataclasses import dataclass


from .utils import (
    convert_name_to_hex, 
    get_new_caption_with_special_token, 
)

FONT_FILE_PATH = '/pyy/openseg_blob/zhanhao/code_arranged/aigc/llm/glyph_clip/fonts'

@dataclass
class TextInstance:
    _font_mapping = None

    text: str
    width: float
    height: float
    left: float
    top: float
    angle: float
    font: str
    color: str = None

    def as_tuple(self):
        return (self.text, (self.left, self.top, self.width, self.height, self.angle))
    
    def get_font_file(self):
        font_id, font_index = self.font.split(',')
        return f"{FONT_FILE_PATH}/{font_id}-{font_index}.ttf"

def get_multiline_text_autowrap(text, font, bbox):
    max_width = bbox[2]
    lines = []
    words = text.split()
    while words:
        line = ''  
        line += (words.pop(0) + ' ')
        while (words and font.getlength(line + words[0]) <= max_width):  
            line += (words.pop(0) + ' ')
        lines.append(line.strip())  
    return "\n".join(lines)


class CanvaRenderDataset(Dataset):
    def __init__(self,
                resolution=1024,
                proportion_empty_prompts=0.1,
                img_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_render_illustration_clean_0104/',
                ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/canva_font_byt5_362k_bbox.json',
                font_ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/font_idx_512.json',
                color_ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/color_idx.json',
                prompt_json_path='/pyy/openseg_blob/weicong/big_file/data/canva-data/canva-llava1.5-bg/',
                auto_wrap=True,
                random_font=True,
                random_color=True,
                load_img_method='old_bg',
                filter_not_text=True,
        ):
        self.resolution = resolution
        self.img_path = img_path
        self.prompt_json_path = prompt_json_path
        self.proportion_empty_prompts = proportion_empty_prompts
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        with open(ann_path, 'r') as f:
            self.ann_list = json.load(f)

        font_path = '/pyy/openseg_blob/weicong/big_file/data/canva-data/font-mapping.json'
        with open(font_path, 'r') as f:
            self.font_dict = json.load(f)
        with open(font_ann_path, 'r') as f:
            self.font_idx_dict = json.load(f)
            self.font_idx_list = list(self.font_idx_dict.items())
            self.reverse_font_dict = {}
            for key in self.font_dict:
                self.reverse_font_dict[self.font_dict[key]] = key
        with open(color_ann_path, 'r') as f:
            self.color_idx_dict = json.load(f)
            self.color_idx_list = list(self.color_idx_dict.items())
        self.auto_wrap = auto_wrap
        self.random_font = random_font
        self.random_color = random_color
        self.filter_not_text = filter_not_text
        assert load_img_method in ('clean_bg', 'old_bg')
        self.load_img_method = load_img_method

    @staticmethod
    def render_text(image: Image, text_instance: TextInstance, resolution: int = 224, auto_wrap: bool = False):
        text, bbox = text_instance.as_tuple()
        bbox = list(map(lambda x: x * resolution / 1645, bbox))
        font_file = text_instance.get_font_file()

        # Binary search for the largest font size that fits the bbox
        font_size = 1
        font_size_upper_bound = 200
        font_size_lower_bound = 1
        draw = ImageDraw.Draw(image)
        font_x, font_y = 0, 0
        while font_size_lower_bound < font_size_upper_bound:
            try:
                font = ImageFont.truetype(font_file, size=font_size)
            except:
                # counter[font_file] += 1
                # print(counter)
                font_list = font_file.split('-')
                prefix = "".join(font_list[:-1])
                font_file = f"{prefix}-0.ttf"
                font = ImageFont.truetype(font_file, size=font_size)
            try:
                # left, top, right, bottom = draw.multiline_textbbox(bbox[:2], text, font=font, align="center")
                if auto_wrap:
                    text = get_multiline_text_autowrap(text, font, bbox)
                    x_offset, y_offset = font.getbbox(text)[:2]
                    x, y = bbox[0] - x_offset, bbox[1] - y_offset
                    left, top, right, bottom = draw.multiline_textbbox((x, y), text, font=font, align="center")
                else:
                    x_offset, y_offset = font.getbbox(text)[:2]
                    x, y = bbox[0] - x_offset, bbox[1] - y_offset
                    left, top, right, bottom = draw.multiline_textbbox((x, y), text, font=font, align="center")
            except:
                # print(f"warning, multiline, {font_file}")
                return font_size
            
            width = right - left
            height = bottom - top

            if width > bbox[2] or height > bbox[3]:
                # font_size_upper_bound = font_size
                font_size_upper_bound = font_size - 1
            else:
                # font_size_lower_bound = font_size + 1
                font_size_lower_bound = font_size
                font_x, font_y = width, height
            # font_size = (font_size_lower_bound + font_size_upper_bound) // 2
            font_size = (font_size_lower_bound + font_size_upper_bound + 1) // 2

        try:    
            font = ImageFont.truetype(font_file, size=font_size)
        except:
            # counter[font_file] += 1
            # print(counter)
            font_list = font_file.split('-')
            prefix = "".join(font_list[:-1])
            font_file = f"{prefix}-0.ttf"
            font = ImageFont.truetype(font_file, size=font_size)

        fill_color = text_instance.color
        x_offset, y_offset = font.getbbox(text)[:2]
        x, y = bbox[0] - x_offset, bbox[1] - y_offset
        if auto_wrap:
            text = get_multiline_text_autowrap(text, font, bbox)
        draw.multiline_text(
            (x + (bbox[2] - font_x) // 2, y + (bbox[3] - font_y) // 2),
            text,
            font=font,
            fill=fill_color,
            align="center",
        )
        return font_size

    def clamp_len(self, l, max_len):
        return max(0, min(l, max_len))

    def check_box(self, bbox, canvas_len=1645):
        bbox[0] = self.clamp_len(bbox[0], canvas_len)
        bbox[1] = self.clamp_len(bbox[1], canvas_len)
        bbox[2] = self.clamp_len(bbox[2], canvas_len - bbox[0])
        bbox[3] = self.clamp_len(bbox[3], canvas_len - bbox[1])
        return bbox

    def get_caption_and_img(self, texts, styles, bboxes):
        font_dict = self.font_idx_dict
        color_dict = self.color_idx_dict

        caption = get_new_caption_with_special_token(texts, styles, self.font_dict, font_idx_dict=font_dict, color_idx_dict=color_dict)
        img = Image.new("RGB", (max(224, self.resolution), max(224, self.resolution)), (0, 0, 0))
        font_sizes = []
        for text, style, bbox in zip(texts, styles, bboxes):
            bbox = self.check_box(bbox)
            text_instance = TextInstance(
                text = text,
                left = bbox[0],
                top = bbox[1],
                width = bbox[2],
                height = bbox[3],
                angle = bbox[4],
                font = style.get("font-family", 'YAFdJrHPPtU,0'),
                color = style.get("color", None)
            )
            font_size = self.render_text(img, text_instance, resolution=max(224, self.resolution), auto_wrap=self.auto_wrap)
            font_sizes.append(font_size)
        return caption, img, font_sizes

    def __len__(self):
        return len(self.ann_list)

    def render_img(self, ann):
        texts = ann['texts']
        styles = ann['styles']
        bboxes = ann['bbox']

        prompt, img, _ = self.get_caption_and_img(texts, styles, bboxes)

        # print(ann['styles'])
        # print(ann['texts'])
        # print(ann['bbox'])
        return img

    def get_random_color(self):
        color_name, _ = random.choice(self.color_idx_list)
        return color_name

    def get_random_font(self):
        font_name, _ = random.choice(self.font_idx_list)
        return font_name

    def get_random_string(self, length):
        # Define the characters that can be used in the string
        characters = string.ascii_letters + string.digits
        # Generate a random string
        random_string = ''.join(random.choice(characters) for i in range(length))
        return random_string

    def __getitem__(self, idx):
        try:
            ann = self.ann_list[idx]
            if self.filter_not_text:
                if len(ann['texts']) == 0:
                    raise ValueError()
            folder = ann['_id'] # ann['rendered_folder']
            if self.load_img_method == 'clean_bg':
                image = Image.open(osp.join(
                    self.img_path, 
                    f'{folder}-bg.png', 
                )).convert('RGB')
            else:
                image = Image.open(osp.join(
                    self.img_path, 
                    folder,
                    't=false.png', 
                )).convert('RGB')

            texts = copy.deepcopy(ann['texts'])
            styles = copy.deepcopy(ann['styles'])
            bboxes = copy.deepcopy(ann['bbox'])
            
            bg_prompt_path = osp.join(self.prompt_json_path, f'{folder}.json')
            with open(bg_prompt_path, 'r') as f:
                bg_prompt = json.load(f)['bg_caption']

            assert len(texts) == len(styles) and len(texts) == len(bboxes)

            if self.random_color:
                for i in range(len(styles)):
                    color_name = self.get_random_color()
                    styles[i]['color'] = convert_name_to_hex(color_name)
            if self.random_font:
                for i in range(len(styles)):
                    # FIXME
                    font_name = self.get_random_font()
                    font_code = self.reverse_font_dict[font_name]
                    styles[i]['font-family'] = font_code
                    styles[i]['font-id'] = font_code.split(',')[0]

            _, img, _ = self.get_caption_and_img(texts, styles, bboxes)

            text_prompt = get_new_caption_with_special_token(texts, styles, self.font_dict, font_idx_dict=self.font_idx_dict, color_idx_dict=self.color_idx_dict)
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))

        for i in range(len(styles)):
            if 'color' in styles[i]:
                styles[i]['color'] = '#ffffff'
        _, mask, _ = self.get_caption_and_img(texts, styles, bboxes)
        mask = mask.convert('L')
                
        new_width, new_height = self.resolution, self.resolution
        image = image.resize((new_width, new_height))
        composed_img = Image.composite(img, image, mask)
        ori_size = (new_height, new_width)
        target_size = (new_height, new_width)
        c_crop = (0, 0)
        
        composed_img = self.transforms(composed_img)

        if "category" in ann:
            bg_prompt = ann["category"] + ". " + bg_prompt
        if "tags" in ann:
            bg_prompt += " Tags: " + ', '.join(ann["tags"])

        random_p = random.random()
        if random_p < self.proportion_empty_prompts:
            bg_prompt = ''

        return {
            'img': composed_img,
            'text_prompt': text_prompt,
            'bg_prompt': bg_prompt,
            'ori_size': ori_size,
            'c_crop': c_crop,
            'target_size': target_size,
            'pad_mask': None,
        }
