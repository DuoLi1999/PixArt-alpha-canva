import json
import os.path as osp
import random
from transformers import AutoTokenizer
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from diffusion.model.t5 import T5Embedder
# from builder import DATASETS
import torch
import torchvision
import torchvision.transforms.functional as F

from PIL import Image

def _resize_and_center_pad(img, resolution, transforms, padding_mode='constant'):
    w, h = img.size
    resize_w, resize_h = _get_resize_shape(w, h, resolution)
    mask = torch.zeros((resolution, resolution))
    padding_values = ((resolution - resize_w) // 2, (resolution - resize_h) // 2, resolution - resize_w - (resolution - resize_w) // 2, resolution - resize_h - (resolution - resize_h) // 2)
    pad_transform = torchvision.transforms.Pad(padding_values, padding_mode=padding_mode)
    mask[(resolution - resize_h) // 2: resize_h + (resolution - resize_h) // 2, (resolution - resize_w) // 2 : resize_w + (resolution - resize_w) // 2] = 1
    return transforms(pad_transform(img.resize((resize_w, resize_h), resample=Image.BICUBIC))), mask

def _get_resize_shape(w, h, resolution):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    rw, rh = resolution[0], resolution[1]
    if w * rh > h * rw:
        return rw, int(rw / (w / h) + 0.5)
    else:
        return int(rh / (h / w) + 0.5), rh

from mmengine import Registry, build_from_cfg

DATASETS = Registry('dataset')
COLLATE_FN = Registry('collate_fn')

def build_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)

# def get_collate_fn(collate_fn_name):
#     return COLLATE_FN.get(collate_fn_name)
@DATASETS.register_module()
class Canva8ChannelsDataset(Dataset):
    def __init__(self,
                resolution=256,
                proportion_empty_prompts=0.0,
                use_embed=True,
                prompt_with_text=True,
                img_path='/home/ld/Project/PixArt-alpha/openseg_blob/weicong/big_file/data/canva-data/canva-render-10.19/',
                ann_path='/home/ld/Project/PixArt-alpha/openseg_blob/weicong/big_file/data/canva-data/canva-binmask/canva-binmask-index.json',
                embed_dir='/home/ld/Project/PixArt-alpha/openseg_blob/weicong/big_file/data/canva-data/canva-binmask-t5-a100/',
                category='all',
                img_type='img', # all(8), aux(8+3), img(3), bg(3)
                target_img_type=None,
                prompt_mode = 'whole',
                train="train"
        ):
        self.resolution = resolution
        self.img_path = img_path
        self.embed_dir = embed_dir
        self.use_embed = use_embed
        self.img_type = img_type
        self.target_img_type = target_img_type
        self.prompt_with_text = prompt_with_text
        self.prompt_mode = prompt_mode
        self.proportion_empty_prompts = proportion_empty_prompts
        self.tokenizer =  AutoTokenizer.from_pretrained('/pyy/yuyang_blob/pyy/code/PixArt-alpha-canva/output/pretrained_models/t5_ckpts/t5-v1_1-xxl')

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        self.ann_list = []
        if category == 'all':
            with open(ann_path, 'r') as f:
                for i, line in enumerate((f.readlines())):
                    self.ann_list.append(json.loads(line))
        else:
            with open(ann_path, 'r') as f:
                self.ann_list = json.load(f)
        if train=="train":
            self.ann_list = self.ann_list[:-50000]
        elif train=='val':
            self.ann_list = self.ann_list[-50000:]


    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        try:
            ann = self.ann_list[idx]
            prompt = ann['caption']
            folder = ann['_id'] # ann['rendered_folder']
            image = Image.open(osp.join(self.img_path, folder, 't=true.png')).convert('RGB')
            w, h = image.size

            if self.use_embed:
                embed_path = osp.join(self.embed_dir, ann['_id'] + '.npy')
                text_embed = np.load(embed_path)
                text_embed = torch.from_numpy(text_embed).squeeze(0)
            else:
                text_embed = None
        except:
            return self.__getitem__(np.random.randint(0, self.__len__()))

        whole_img_tensor, _ = _resize_and_center_pad(image, self.resolution, self.transforms, padding_mode='edge')

        alpha_tensor = whole_img_tensor

        if self.prompt_mode == 'whole':
            if "category" in ann:
                prompt = ann["category"] + ". " + prompt
            if self.prompt_with_text and "texts" in ann:
                texts = [f'"{text}"' for text in ann["texts"]]
                prompt += " Text: " + ', '.join(texts)
            if "tags" in ann:
                prompt += " Tags: " + ', '.join(ann["tags"])
        elif self.prompt_mode == 'textlist':
            textlist = [text if isinstance(text, str) else text[0] for text in ann["texts"]]
            prompt = ', '.join(textlist)

        if random.random() < self.proportion_empty_prompts:
            prompt = ''
            if self.use_embed:
                embed_path = osp.join(self.embed_dir, 'empty.npy')
                text_embed = np.load(embed_path)
                text_embed = torch.from_numpy(text_embed).squeeze(0)
        text_tokens_and_mask = self.tokenizer(
            prompt,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        # if text_embed[-1].equal(text_embed[-2]):
        #     mask=torch.ones(text_embed.size(0),dtype=torch.bool)

        # else:
        #     mask=~torch.all(text_embed-text_embed[-1]==0,dim=-1)
        mask=text_tokens_and_mask['attention_mask']>0
        return alpha_tensor,text_embed,mask.squeeze(0),False
        