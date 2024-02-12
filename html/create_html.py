import json
from collections import Counter
import random
import webcolors
from design_diff.utils import get_newbench_prompt, get_crello_prompt


index_file = "/mnt/openseg_blob/liuzeyu/datasets2/canva_poster_filtered.json"
index_file = "/openseg_blob/liuzeyu/datasets2/canva_illustration.json"
# caption_dir = "/mnt/openseg_blob/weicong/big_file/data/canva-data/metadata/"
json_path = '/openseg_blob/weicong/big_file/data/canva-data/canva_benchmark_chosen_meta/'
font_path = '/openseg_blob/weicong/big_file/data/canva-data/font-mapping.json'

def get_caption(meta):
    prompt = meta['caption'] # ann['llava_llama2_caption']
    if "category" in meta:
        prompt = meta["category"] + ". " + prompt
    if "texts" in meta:
        texts = [f'"{text}"' for text in meta["texts"]]
        prompt += " Text: " + ', '.join(texts)
    if "tags" in meta:
        prompt += " Tags: " + ', '.join(meta["tags"])
    # if "texts" in meta:
    #     texts = [f'"{text}"' for text in meta["texts"]]
    #     prompt = ', '.join(texts)
    # else:
    #     prompt = ""
    return prompt

def get_caption_bg(meta, json_path):
    caption_path = os.path.join(json_path, f"{meta['_id']}.json")
    with open(caption_path, 'r') as f:
        caption_ann = json.load(f)
    prompt = caption_ann['caption']
    if "category" in meta:
        prompt = meta["category"] + ". " + prompt
    if "tags" in meta:
        prompt += " Tags: " + ', '.join(meta["tags"])
    return prompt

def get_caption_text(meta, font_dict, use_style=False):
    prompt = ""
    if use_style:
        prompt = "</s>"
        for text, style in zip(meta["texts"], meta["styles"]):
            text_prompt = f'Text: {text}.'
            if 'color' in style:
                hex_color = style["color"]  
                rgb_color = webcolors.hex_to_rgb(hex_color)  
                # get color name  
                color_name = convert_rgb_to_names(rgb_color)  
                text_prompt += f' Color: {color_name}.'
            if 'font-family' in style and style["font-family"] in font_dict:
                font_name = font_dict[style["font-family"]]
                text_prompt += f' Font family: {font_name}.'
            if 'font-weight' in style:
                text_prompt += f' Font weight: {style["font-weight"]}.'
            if 'font-size' in style:
                text_prompt += f' Font size: {style["font-size"]}.'
            if 'text-align' in style:
                text_prompt += f' Text align: {style["text-align"]}.'
            text_prompt = '"' + text_prompt + '"</s>'
            prompt = prompt + text_prompt
    else:
        for text in meta["texts"]:
            text_prompt = f'Text: {text}.'
            text_prompt = '"' + text_prompt + '". '
            prompt = prompt + text_prompt
    return prompt

def closest_color(requested_color):  
    min_colors = {}  
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():  
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)  
        rd = (r_c - requested_color[0]) ** 2  
        gd = (g_c - requested_color[1]) ** 2  
        bd = (b_c - requested_color[2]) ** 2  
        min_colors[(rd + gd + bd)] = name  
    return min_colors[min(min_colors.keys())]

def convert_rgb_to_names(rgb_tuple):  
    try:  
        color_name = webcolors.rgb_to_name(rgb_tuple)  
    except ValueError:  
        color_name = closest_color(rgb_tuple)  
    return color_name

ann_path = 'openseg_blob/liuzeyu/datasets2/test_font_byt5_100.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/typo_clip_data/typoclip_onebox_200-400_simplefont_100k_val.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/typo_clip_data/typoclip_onebox_200-400_512font_100k_val.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/test_font_byt5_100_color_and_font.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/typo_clip_data/typoclip_onebox_200-400_simplefont_100k_multibox_val.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/canva_ann_json/canva_font_byt5_400_val.json'
with open(ann_path, 'r') as f:
    data = json.load(f)
# data = data[:100]

# ann_path = '/openseg_blob/liuzeyu/datasets2/canva_ann_json/font_30k_subset.json'
# with open(ann_path, 'r') as f:
#     ann_list = json.load(f)
# data = [ann_list[i] for i in range(0, 30000, 300)]


print("""
    <html>
       <head>
       <style>
       .row {
           display: flex;  
           flex-direction: row;  
           justify-content: start;  
           align-items: center;  
           margin-bottom: 16px; 
       }
       .column {
           flex: 15%;
           padding: 0 16px;
       }
       </style>
       </head>
       <body>
""")

# data = [x for x in data if "Illustration" in x['tags'] or 'illustration' in x['tags']]

# print(f"""
#         <div class="row">
#             <p> Col1: gt </p>
#         </div>
# """)
# print(f"""
#         <div class="row">
#             <p> Col1: gt </p>
#         </div>
# """)
print(f"""
        <div class="row">
            <p> Col1: before tune multibox text </p>
        </div>
""")
print(f"""
        <div class="row">
            <p> Col2: after tune multibox text </p>
        </div>
""")

with open(font_path, 'r') as f:
    font_dict = json.load(f)

print(f"<p>{len(data)}</p>")
folder_list = [
    # '1201_gt',
    # '1201_glyph_condition_byt5_noise0x05_lr1e-4_nonlinear',
    # '1207_stage1_byt5_small_encoder_font_test_glyph',

    # '1221_stage1_byt5_small_30k_clip_unfreeze_encoder_noise0x05_nonlinear_font_test_glyph',

    # '0117_densetext_50-100_gt',
    # '0117_stage2_362k_20ep',
    # '0115_densetext_gt',
    # '0117_densetext_stage2_362k_20ep',
    '0130_densetext_font512_stage2_densetext',
]

for index, item in enumerate(data):
    file_prefix = f"https://openseg.blob.core.windows.net/openseg-aml/liuzeyu/diffusion_design_if/DesignDiff/eval/"
    file_sufix = "?sv=2021-10-04&st=2023-11-28T03%3A59%3A26Z&se=2024-11-29T03%3A59%3A00Z&sr=c&sp=rl&sig=kgIQ6YOdgXRxY9kO4grRT4Rx2YRZFHv3V3XVauteVkM%3D"
    if index % 1 == 0:
        print('<div class="row">')
    for i, folder in enumerate(folder_list):
        # if i == 0:
        #     img_name = f"/gt_font_{index}.png"
        #     print(f"""
        #             <div class="column">
        #                 <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             </div>""")
        # else:
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                </div>""")
    if index % 1 == 0:
        print('</div>')
    print(f"""
            <div class="row">
                <p>Index: {index} <br>
                {item['texts']} </p>
            </div>
    """)
    # print(f"""
    #         <div class="row">
    #             <p>Index: {index} <br>
    #             Category: {item["category"]} <br>
    #             {item["BGCaption_llava"]} <br>
    #             Keywords: {item["keywords"]}</p>
    #         </div>
    # """)
