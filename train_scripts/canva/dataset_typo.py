import json
import os.path as osp
import random
import copy

import numpy as np
from PIL import Image

import torch

from .utils import (
    convert_name_to_hex, 
    get_new_caption_with_special_token, 
)

from .canva_render_dataset import CanvaRenderDataset

FONT_FILE_PATH = '/pyy/openseg_blob/zhanhao/code_arranged/aigc/llm/glyph_clip/fonts'

class CanvaRenderWithMaskDataset(CanvaRenderDataset):
    def __init__(self,
            resolution=512,
            proportion_empty_prompts=0.1,
            img_path='/pyy/openseg_blob/weicong/big_file/data/canva-data/canva-render-10.19/',
            ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_font_byt5_nms_filter_512_fonts_100k_train.json',
            font_ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/font_idx_512.json',
            color_ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/color_idx.json',
            prompt_json_path='/pyy/openseg_blob/weicong/big_file/data/canva-data/canva-llava1.5-bg/',
            auto_wrap=True,
            random_font=True,
            random_color=True,
            text_feat_length=512,
            load_img_method='old_bg',
            filter_not_text=True,
        ):
        super().__init__(
            resolution=resolution,
            proportion_empty_prompts=proportion_empty_prompts,
            img_path=img_path,
            ann_path=ann_path,
            font_ann_path=font_ann_path,
            color_ann_path=color_ann_path,
            prompt_json_path=prompt_json_path,
            auto_wrap=auto_wrap,
            random_font=random_font,
            random_color=random_color,
            load_img_method=load_img_method,
            filter_not_text=filter_not_text,
        )
        self.text_feat_length = text_feat_length
    
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
        
        composed_img = self.transforms(composed_img)
        
        glyph_attn_mask = self.get_glyph_attn_mask(texts, bboxes)

        if "category" in ann:
            bg_prompt = ann["category"] + ". " + bg_prompt
        if "tags" in ann:
            bg_prompt += " Tags: " + ', '.join(ann["tags"])

        random_p = random.random()
        if random_p < self.proportion_empty_prompts:
            bg_prompt = ''
            text_prompt = ''

        return {
            'img': composed_img,
            'text_prompt': text_prompt,
            'bg_prompt': bg_prompt,
            'pad_mask': None,
            'glyph_attn_mask': glyph_attn_mask
        }

    def get_glyph_attn_mask(self, texts, bboxes):
        text_idx_list = self.get_text_start_pos(texts)
        mask_tensor = torch.zeros(
            self.resolution // 8, self.resolution // 8, self.text_feat_length,
        )
        for idx, bbox in enumerate(bboxes):
            # box is in [x, y, w, h, angle] format
            # area of [y:y+h, x:x+w]
            bbox = [int(v / 1645 * (self.resolution // 8) + 0.5) for v in bbox]
            bbox[2] = max(bbox[2], 1)
            bbox[3] = max(bbox[3], 1)
            bbox[0: 2] =  np.clip(bbox[0: 2], 0, self.resolution // 8 - 1).tolist()
            bbox[2: 4] = np.clip(bbox[2: 4], 1,self.resolution // 8).tolist()
            mask_tensor[
                bbox[1]: bbox[1] + bbox[3], 
                bbox[0]: bbox[0] + bbox[2], 
                text_idx_list[idx]: text_idx_list[idx + 1]
            ] = 1
        return mask_tensor

    def get_text_start_pos(self, texts):
        prompt = ""
        '''
        Text "{text}" in {color}, {type}, {weight}.
        '''
        pos_list = []
        for text in texts:
            pos_list.append(len(prompt))
            text_prompt = f'Text "{text}"'

            attr_list = ['0', '1']

            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
            text_prompt += ". "

            prompt = prompt + text_prompt
        pos_list.append(len(prompt))
        return pos_list