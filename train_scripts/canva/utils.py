import torch
import torchvision
import webcolors

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

def convert_name_to_hex(color_name):
    hex_value = webcolors.name_to_hex(color_name)
    return hex_value

def get_special_token_text(meta, font_dict, font_idx_dict, color_idx_dict):
    prompt = ""
    '''
    Text "{text}" in {color}, {type}.
    '''
    for text, style in zip(meta["texts"], meta["styles"]):
        text_prompt = f'Text "{text}"'

        attr_list = []
        if 'color' in style:
            hex_color = style["color"]  
            rgb_color = webcolors.hex_to_rgb(hex_color)  
            # get color name  
            color_name = convert_rgb_to_names(rgb_color)
            if color_name in color_idx_dict:
                attr_list.append(f"<color-{color_idx_dict[color_name]}>")
        if 'font-family' in style and style["font-family"] in font_dict:
            font_name = font_dict[style["font-family"]]
            if font_name in font_idx_dict:
                attr_list.append(f"<font-{font_idx_dict[font_name]}>")

        if len(attr_list) > 0:
            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
        text_prompt += ". "

        prompt = prompt + text_prompt

    return prompt

def get_new_caption_text(texts, styles, font_dict):
    prompt = ""
    '''
    Text "{text}" in {color}, {type}, {weight}.
    '''
    for text, style in zip(texts, styles):
        text_prompt = f'Text "{text}"'

        attr_list = []
        if 'color' in style:
            hex_color = style["color"]  
            rgb_color = webcolors.hex_to_rgb(hex_color)  
            # get color name  
            color_name = convert_rgb_to_names(rgb_color)  
            attr_list.append(color_name)
        if 'font-family' in style and style["font-family"] in font_dict:
            font_name = font_dict[style["font-family"]]
            attr_list.append(font_name)
        if 'font-weight' in style:
            attr_list.append(style["font-weight"])

        if len(attr_list) > 0:
            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
        text_prompt += ". "

        prompt = prompt + text_prompt
    return prompt

def get_new_caption_with_special_token(texts, styles, font_dict, font_idx_dict=None, color_idx_dict=None):
    prompt = ""
    '''
    Text "{text}" in {color}, {type}, {weight}.
    '''
    for text, style in zip(texts, styles):
        text_prompt = f'Text "{text}"'

        attr_list = []
        if 'color' in style:
            hex_color = style["color"]  
            rgb_color = webcolors.hex_to_rgb(hex_color)
            # get color name  
            color_name = convert_rgb_to_names(rgb_color)
            if color_idx_dict is not None:
                attr_list.append(f"<color-{color_idx_dict[color_name]}>")
            else:
                attr_list.append(color_name)
        if 'font-family' in style and style["font-family"] in font_dict:
            font_name = font_dict[style["font-family"]]
            if font_idx_dict is not None and font_name in font_idx_dict:
                attr_list.append(f"<font-{font_idx_dict[font_name]}>")
            else:
                attr_list.append(font_name)

        if len(attr_list) > 0:
            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
        text_prompt += ". "

        prompt = prompt + text_prompt
    return prompt


def get_new_caption_with_start_end_token(texts, styles, font_dict, text_start_spe_token, text_end_spe_token):
    prompt = ""
    '''
    <start>Text "{text}" in {color}, {type}, {weight}. <end>
    '''
    for text, style in zip(texts, styles):
        text_prompt = f'{text_start_spe_token}Text "{text}"'

        attr_list = []
        if 'color' in style:
            hex_color = style["color"]  
            rgb_color = webcolors.hex_to_rgb(hex_color)
            # get color name  
            color_name = convert_rgb_to_names(rgb_color)
            attr_list.append(color_name)
        if 'font-family' in style and style["font-family"] in font_dict:
            font_name = font_dict[style["font-family"]]
            attr_list.append(font_name)

        if len(attr_list) > 0:
            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
        text_prompt += f". {text_end_spe_token}"

        prompt = prompt + text_prompt
    return prompt
