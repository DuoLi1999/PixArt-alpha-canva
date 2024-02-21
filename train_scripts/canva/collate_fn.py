import torch
import numpy as np

def custom_collate_fn_sdxl(
    batched_data, 
    tokenizers,
    byt5_tokenizer=None,
    byt5_max_length=None):    
    # imgs are already resized and cropped
    imgs = [item['img'] for item in batched_data]
    prompts = [item['prompt'] for item in batched_data]
    text_prompts = [item['text_prompt'] for item in batched_data]
    valid_text_prompts = [0 if item['text_prompt'] == "" else 1 for item in batched_data]
    ori_sizes = [item['ori_size'] for item in batched_data]
    c_crops = [item['c_crop'] for item in batched_data]
    target_sizes = [item['target_size'] for item in batched_data]
    pad_masks = [item['pad_mask'] for item in batched_data]
    
    # b,c,h,w
    imgs = torch.stack(imgs, dim=0)
    # b,1,h,w
    pad_masks = torch.stack(pad_masks, dim=0).unsqueeze(1)
    
    # Compute text token id
    # each entry in text_input_id_batchs is a 
    # batched tokenization results of one tokenizer
    text_input_id_batchs = []
    for tokenizer in tokenizers:
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # Tensor with size (batch_size, seq_len)
        text_input_ids = text_inputs.input_ids
        text_input_id_batchs.append(text_input_ids)
    
    if byt5_tokenizer is not None:
        byt5_text_inputs = byt5_tokenizer(
            text_prompts,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # b, byt5_max_length
        byt5_text_input_ids = byt5_text_inputs.input_ids
        # b, byt5_max_length
        byt5_attn_mask = byt5_text_inputs.attention_mask
    else:
        byt5_text_input_ids = None
        byt5_attn_mask = None
    
    # B, 1, 1
    valid_text_prompts = torch.tensor(valid_text_prompts)[:, None, None]
    
    # Additional embeddings needed for SDXL UNet
    add_time_ids = [list(ori_size + c_crop + target_size) 
                    for ori_size, c_crop, target_size 
                    in zip(ori_sizes, c_crops, target_sizes)]
    # batch_size, 6
    add_time_ids = torch.tensor(add_time_ids)
        
    return dict(imgs=imgs,
                pad_masks=pad_masks,
                prompts=prompts,
                text_prompts=text_prompts,
                time_ids=add_time_ids,
                text_input_id_batchs=text_input_id_batchs,
                byt5_text_input_ids=byt5_text_input_ids,
                byt5_attn_mask=byt5_attn_mask,
                valid_text_prompts=valid_text_prompts,
            )


def custom_collate_fn_sdxl_long_prompt(
    batched_data, 
    tokenizers,
    byt5_tokenizer=None,
    byt5_max_length=None):    
    # imgs are already resized and cropped
    imgs = [item['img'] for item in batched_data]
    prompts = [item['prompt'] for item in batched_data]
    text_prompts = [item['text_prompt'] for item in batched_data]
    valid_text_prompts = [0 if item['text_prompt'] == "" else 1 for item in batched_data]
    ori_sizes = [item['ori_size'] for item in batched_data]
    c_crops = [item['c_crop'] for item in batched_data]
    target_sizes = [item['target_size'] for item in batched_data]
    pad_masks = [item['pad_mask'] for item in batched_data]
    
    # b,c,h,w
    imgs = torch.stack(imgs, dim=0)
    # b,1,h,w
    if pad_masks[0] is not None:
        pad_masks = torch.stack(pad_masks, dim=0).unsqueeze(1)
    else:
        pad_masks = None
    
    # Compute text token id
    # each entry in text_input_id_batchs is a 
    # batched tokenization results of one tokenizer
    
    text_input_id_batchs = []
    for tokenizer in tokenizers:
        pad_token = tokenizer.pad_token_id
        total_tokens = tokenizer(prompts, truncation=False)['input_ids']
        bos = total_tokens[0][0]
        eos = total_tokens[0][-1]
        total_tokens = [i[1:-1] for i in total_tokens]
        new_total_tokens = []
        for token_ids in total_tokens:
            new_total_tokens.append([])
            empty_flag = True
            while len(token_ids) >= 75:
                head_75_tokens = [token_ids.pop(0) for _ in range(75)]
                temp_77_token_ids = [bos] + head_75_tokens + [eos]
                new_total_tokens[-1].append(temp_77_token_ids)
                empty_flag = False
            if len(token_ids) > 0 or empty_flag:
                padding_len = 75 - len(token_ids)
                temp_77_token_ids = [bos] + token_ids + [eos] + [pad_token] * padding_len
                new_total_tokens[-1].append(temp_77_token_ids)
        max_77_len = len(max(new_total_tokens, key=len))
        for new_tokens in new_total_tokens:
            if len(new_tokens) < max_77_len:
                padding_len = max_77_len - len(new_tokens)
                new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b,segment_len,77
        new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
        text_input_id_batchs.append(new_total_tokens)
    if text_input_id_batchs[0].shape[1] > text_input_id_batchs[1].shape[1]:
        tokenizer = tokenizers[1]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = text_input_id_batchs[0].shape[1] - text_input_id_batchs[1].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(text_input_id_batchs[1].shape[0], 1, 1)
        text_input_id_batchs[1] = torch.cat((text_input_id_batchs[1], padding_part), dim=1)
    elif text_input_id_batchs[0].shape[1] < text_input_id_batchs[1].shape[1]:
        tokenizer = tokenizers[0]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = text_input_id_batchs[1].shape[1] - text_input_id_batchs[0].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(text_input_id_batchs[0].shape[0], 1, 1)
        text_input_id_batchs[0] = torch.cat((text_input_id_batchs[0], padding_part), dim=1)        

    
    if byt5_tokenizer is not None:
        byt5_text_inputs = byt5_tokenizer(
            text_prompts,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # b, byt5_max_length
        byt5_text_input_ids = byt5_text_inputs.input_ids
        # b, byt5_max_length
        byt5_attn_mask = byt5_text_inputs.attention_mask
    else:
        byt5_text_input_ids = None
        byt5_attn_mask = None
    
    # B, 1, 1
    valid_text_prompts = torch.tensor(valid_text_prompts)[:, None, None]
    
    # Additional embeddings needed for SDXL UNet
    add_time_ids = [list(ori_size + c_crop + target_size) 
                    for ori_size, c_crop, target_size 
                    in zip(ori_sizes, c_crops, target_sizes)]
    # batch_size, 6
    add_time_ids = torch.tensor(add_time_ids)
        
    return dict(imgs=imgs,
                pad_masks=pad_masks,
                prompts=prompts,
                text_prompts=text_prompts,
                time_ids=add_time_ids,
                text_input_id_batchs=text_input_id_batchs,
                byt5_text_input_ids=byt5_text_input_ids,
                byt5_attn_mask=byt5_attn_mask,
                valid_text_prompts=valid_text_prompts,
            )


def custom_collate_fn_sdxl_long_prompt_canva_render(
    batched_data, 
    tokenizers,
    byt5_tokenizer=None,
    byt5_max_length=None):    
    # imgs are already resized and cropped
    imgs = [item['img'] for item in batched_data]
    text_prompts = [item['text_prompt'] for item in batched_data]
    bg_prompts = [item['bg_prompt'] for item in batched_data]
    valid_bg_prompts = [0 if item['bg_prompt'] == "" else 1 for item in batched_data]
    ori_sizes = [item['ori_size'] for item in batched_data]
    c_crops = [item['c_crop'] for item in batched_data]
    target_sizes = [item['target_size'] for item in batched_data]
    pad_masks = [item['pad_mask'] for item in batched_data]
    if 'glyph_attn_mask' in batched_data[0] and batched_data[0]['glyph_attn_mask'] is not None:
        glyph_attn_masks = [item['glyph_attn_mask'] for item in batched_data]
        bg_attn_masks = [torch.sum(item['glyph_attn_mask'], dim=-1) == 0 for item in batched_data]
        glyph_attn_masks = torch.stack(glyph_attn_masks, dim=0)
        bg_attn_masks = torch.stack(bg_attn_masks, dim=0).to(glyph_attn_masks.dtype)
    else:
        glyph_attn_masks = None
        bg_attn_masks = None
    
    # b,c,h,w
    imgs = torch.stack(imgs, dim=0)
    # b,1,h,w
    if pad_masks[0] is not None:
        pad_masks = torch.stack(pad_masks, dim=0).unsqueeze(1)
    else:
        pad_masks = None
    
    # Compute text token id
    # each entry in text_input_id_batchs is a 
    # batched tokenization results of one tokenizer
    
    bg_prompt_id_batchs = []
    for tokenizer in tokenizers:
        pad_token = tokenizer.pad_token_id
        total_tokens = tokenizer(bg_prompts, truncation=False)['input_ids']
        bos = total_tokens[0][0]
        eos = total_tokens[0][-1]
        total_tokens = [i[1:-1] for i in total_tokens]
        new_total_tokens = []
        for token_ids in total_tokens:
            new_total_tokens.append([])
            empty_flag = True
            while len(token_ids) >= 75:
                head_75_tokens = [token_ids.pop(0) for _ in range(75)]
                temp_77_token_ids = [bos] + head_75_tokens + [eos]
                new_total_tokens[-1].append(temp_77_token_ids)
                empty_flag = False
            if len(token_ids) > 0 or empty_flag:
                padding_len = 75 - len(token_ids)
                temp_77_token_ids = [bos] + token_ids + [eos] + [pad_token] * padding_len
                new_total_tokens[-1].append(temp_77_token_ids)
        max_77_len = len(max(new_total_tokens, key=len))
        for new_tokens in new_total_tokens:
            if len(new_tokens) < max_77_len:
                padding_len = max_77_len - len(new_tokens)
                new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b,segment_len,77
        new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
        bg_prompt_id_batchs.append(new_total_tokens)
    if bg_prompt_id_batchs[0].shape[1] > bg_prompt_id_batchs[1].shape[1]:
        tokenizer = tokenizers[1]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = bg_prompt_id_batchs[0].shape[1] - bg_prompt_id_batchs[1].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(bg_prompt_id_batchs[1].shape[0], 1, 1)
        bg_prompt_id_batchs[1] = torch.cat((bg_prompt_id_batchs[1], padding_part), dim=1)
    elif bg_prompt_id_batchs[0].shape[1] < bg_prompt_id_batchs[1].shape[1]:
        tokenizer = tokenizers[0]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = bg_prompt_id_batchs[1].shape[1] - bg_prompt_id_batchs[0].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(bg_prompt_id_batchs[0].shape[0], 1, 1)
        bg_prompt_id_batchs[0] = torch.cat((bg_prompt_id_batchs[0], padding_part), dim=1)        

    
    byt5_text_inputs = byt5_tokenizer(
        text_prompts,
        padding="max_length",
        max_length=byt5_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    # b, byt5_max_length
    byt5_text_input_ids = byt5_text_inputs.input_ids
    # b, byt5_max_length
    byt5_attn_mask = byt5_text_inputs.attention_mask
    
    # B, 1, 1
    valid_bg_prompts = torch.tensor(valid_bg_prompts)[:, None, None]
    
    # Additional embeddings needed for SDXL UNet
    add_time_ids = [list(ori_size + c_crop + target_size) 
                    for ori_size, c_crop, target_size 
                    in zip(ori_sizes, c_crops, target_sizes)]
    # batch_size, 6
    add_time_ids = torch.tensor(add_time_ids)
        
    return dict(
        imgs=imgs,
        pad_masks=pad_masks,
        bg_prompts=bg_prompts,
        text_prompts=text_prompts,
        time_ids=add_time_ids,
        bg_prompt_id_batchs=bg_prompt_id_batchs,
        valid_bg_prompts=valid_bg_prompts,
        byt5_text_input_ids=byt5_text_input_ids,
        byt5_attn_mask=byt5_attn_mask,
        glyph_attn_masks=glyph_attn_masks,
        bg_attn_masks=bg_attn_masks,
    )


def custom_collate_fn_sdxl_long_prompt_canva_render_split_text(
    batched_data, 
    tokenizers,
    byt5_tokenizer=None,
    byt5_max_length=None,
    clip_text_start_spe_token=None,
    clip_text_end_spe_token=None,
    determine_by_tokenizer_idx=0,
):    
    # imgs are already resized and cropped
    imgs = [item['img'] for item in batched_data]
    text_prompts = [item['text_prompt'] for item in batched_data]
    clip_prompts = [item['clip_prompt'] for item in batched_data]
    ori_sizes = [item['ori_size'] for item in batched_data]
    c_crops = [item['c_crop'] for item in batched_data]
    target_sizes = [item['target_size'] for item in batched_data]
    pad_masks = [item['pad_mask'] for item in batched_data]
    byt5_text_idx_lists = [item['text_idx_list'] for item in batched_data]
    glyph_masks = [item['glyph_mask'] for item in batched_data]
    
    # b,c,h,w
    imgs = torch.stack(imgs, dim=0)
    # b,1,h,w
    if pad_masks[0] is not None:
        pad_masks = torch.stack(pad_masks, dim=0).unsqueeze(1)
    else:
        pad_masks = None
    
    # Compute text token id
    # each entry in text_input_id_batchs is a 
    # batched tokenization results of one tokenizer
    clip_prompt_id_batchs = [None, None]
    
    clip_text_idx_lists = []
    
    tokenizer = tokenizers[determine_by_tokenizer_idx]
    clip_text_start_spe_token_id = tokenizer.convert_tokens_to_ids(clip_text_start_spe_token)
    clip_text_end_spe_token_id = tokenizer.convert_tokens_to_ids(clip_text_end_spe_token)
    pad_token = tokenizer.pad_token_id
    total_tokens = tokenizer(clip_prompts, truncation=False)['input_ids']
    bos = total_tokens[0][0]
    eos = total_tokens[0][-1]
    total_tokens = [i[1:-1] for i in total_tokens]
    new_total_tokens = []
    for token_ids in total_tokens:
        new_total_tokens.append([])
        clip_text_idx_lists.append([])
        # empty_flag for the case that the token_ids is empty
        empty_flag = True
        while len(token_ids) > 0:
            empty_flag = False
            head_76_tokens = [bos]
            while len(head_76_tokens) < 76 and len(token_ids) > 0:
                next_id = token_ids.pop(0)
                if next_id == clip_text_start_spe_token_id:
                    assert len(clip_text_idx_lists[-1]) == 0 or len(clip_text_idx_lists[-1][-1]) == 2
                    current_idx = (len(new_total_tokens[-1])) * 77 + len(head_76_tokens)
                    clip_text_idx_lists[-1].append([current_idx])
                elif next_id == clip_text_end_spe_token_id:
                    assert len(clip_text_idx_lists[-1][-1]) == 1
                    current_idx = (len(new_total_tokens[-1])) * 77 + len(head_76_tokens)
                    clip_text_idx_lists[-1][-1].append(current_idx)
                else:
                    head_76_tokens.append(next_id)
            if len(head_76_tokens) == 76:
                temp_77_token_ids = head_76_tokens + [eos]
                new_total_tokens[-1].append(temp_77_token_ids)
            elif len(token_ids) == 0:
                padding_len = 76 - len(head_76_tokens)
                temp_77_token_ids = head_76_tokens + [eos] + [pad_token] * padding_len
                new_total_tokens[-1].append(temp_77_token_ids)
        if empty_flag:
            temp_77_token_ids = [bos] + [eos] + [pad_token] * 75
            new_total_tokens[-1].append(temp_77_token_ids)
        for text_segment_idx in range(len(clip_text_idx_lists[-1])):
            clip_text_idx_lists[-1][text_segment_idx] = list(
                range(
                    clip_text_idx_lists[-1][text_segment_idx][0], 
                    clip_text_idx_lists[-1][text_segment_idx][1]
                )
            )
        
    max_77_len = len(max(new_total_tokens, key=len))
    for new_tokens in new_total_tokens:
        if len(new_tokens) < max_77_len:
            padding_len = max_77_len - len(new_tokens)
            new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
    # b,segment_len,77
    new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
    clip_prompt_id_batchs[determine_by_tokenizer_idx] = new_total_tokens

    another_tokenizer_idx = 1 - determine_by_tokenizer_idx
    tokenizer = tokenizers[another_tokenizer_idx]
    pure_clip_prompts = [
        i.replace(clip_text_start_spe_token, '').replace(clip_text_end_spe_token, '') for i in clip_prompts
    ]
    pad_token = tokenizer.pad_token_id
    total_tokens = tokenizer(pure_clip_prompts, truncation=False)['input_ids']
    bos = total_tokens[0][0]
    eos = total_tokens[0][-1]
    total_tokens = [i[1:-1] for i in total_tokens]
    new_total_tokens = []
    for token_ids in total_tokens:
        new_total_tokens.append([])
        # empty_flag for the case that the token_ids is empty
        empty_flag = True
        while len(token_ids) >= 75:
            head_75_tokens = [token_ids.pop(0) for _ in range(75)]
            temp_77_token_ids = [bos] + head_75_tokens + [eos]
            new_total_tokens[-1].append(temp_77_token_ids)
            empty_flag = False                    
        if len(token_ids) > 0 or empty_flag:
            padding_len = 75 - len(token_ids)
            temp_77_token_ids = [bos] + token_ids + [eos] + [pad_token] * padding_len
            new_total_tokens[-1].append(temp_77_token_ids)
    max_77_len = len(max(new_total_tokens, key=len))
    for new_tokens in new_total_tokens:
        if len(new_tokens) < max_77_len:
            padding_len = max_77_len - len(new_tokens)
            new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
    # b,segment_len,77
    new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
    clip_prompt_id_batchs[another_tokenizer_idx] = new_total_tokens
    
    if clip_prompt_id_batchs[0].shape[1] > clip_prompt_id_batchs[1].shape[1]:
        tokenizer = tokenizers[1]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = clip_prompt_id_batchs[0].shape[1] - clip_prompt_id_batchs[1].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(clip_prompt_id_batchs[1].shape[0], 1, 1)
        clip_prompt_id_batchs[1] = torch.cat((clip_prompt_id_batchs[1], padding_part), dim=1)
    elif clip_prompt_id_batchs[0].shape[1] < clip_prompt_id_batchs[1].shape[1]:
        tokenizer = tokenizers[0]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = clip_prompt_id_batchs[1].shape[1] - clip_prompt_id_batchs[0].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(clip_prompt_id_batchs[0].shape[0], 1, 1)
        clip_prompt_id_batchs[0] = torch.cat((clip_prompt_id_batchs[0], padding_part), dim=1)              
    
    byt5_text_inputs = byt5_tokenizer(
        text_prompts,
        padding="max_length",
        max_length=byt5_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    # b, byt5_max_length
    byt5_text_input_ids = byt5_text_inputs.input_ids
    # b, byt5_max_length
    byt5_attn_mask = byt5_text_inputs.attention_mask
    
    
    # Additional embeddings needed for SDXL UNet
    add_time_ids = [list(ori_size + c_crop + target_size) 
                    for ori_size, c_crop, target_size 
                    in zip(ori_sizes, c_crops, target_sizes)]
    # batch_size, 6
    add_time_ids = torch.tensor(add_time_ids)
    
    #b,h,w->b,1,h,w
    glyph_masks = torch.stack(glyph_masks, dim=0).unsqueeze(1)
    
    # clip_text_idx_lists: List[List[List[idx1, idx2, idx3,...], List[idx1, idx2,...],...], ...]
    # clip_text_feat_idx: List[List[idx1,idx2,idx3,idx4...], ...]
    clip_text_feat_idx = [sum(i, []) for i in clip_text_idx_lists] 
    clip_max_text_feat_len = len(max(clip_text_feat_idx, key=len))

    refill_indexes_batch = []
    refill_indexes_length = []
    for batch_idx, item in enumerate(clip_text_feat_idx):
        refill_indexes_length.append(torch.tensor(item, dtype=torch.long))
        refill_indexes_batch.append(torch.ones(len(item), dtype=torch.long) * batch_idx)
    refill_idx = [torch.cat(refill_indexes_batch), torch.cat(refill_indexes_length)]

    self_attn_mask = torch.ones(
        len(clip_text_feat_idx),
        clip_max_text_feat_len,
        clip_max_text_feat_len,
    )
    for idx_list_idx in range(len(clip_text_feat_idx)):
        if len(clip_text_feat_idx[idx_list_idx]) < clip_max_text_feat_len:
            pad_len = clip_max_text_feat_len - len(clip_text_feat_idx[idx_list_idx])
            clip_text_feat_idx[idx_list_idx] = (
                clip_text_feat_idx[idx_list_idx] + 
                [clip_prompt_id_batchs[0].shape[1] * clip_prompt_id_batchs[0].shape[2]] * pad_len
            )
            self_attn_mask[idx_list_idx, :, -pad_len:] = 0
    # b, clip_text_max_len,1
    clip_text_feat_idx = torch.tensor(clip_text_feat_idx).unsqueeze(-1)
    
    byt5_text_feat_idx = [sum(i, []) for i in byt5_text_idx_lists]
    byt5_max_text_feat_len = len(max(byt5_text_feat_idx, key=len))
    
    for idx_list_idx in range(len(byt5_text_feat_idx)):
        if len(byt5_text_feat_idx[idx_list_idx]) < byt5_max_text_feat_len:
            byt5_text_feat_idx[idx_list_idx] = (
                byt5_text_feat_idx[idx_list_idx] + 
                [byt5_max_length] * 
                (byt5_max_text_feat_len - len(byt5_text_feat_idx[idx_list_idx]))
            )
    # b,byt5_text_max_len,1
    byt5_text_feat_idx = torch.tensor(byt5_text_feat_idx).unsqueeze(-1)

    cross_attn_mask = torch.zeros(
        len(clip_text_feat_idx),
        clip_max_text_feat_len,
        byt5_max_text_feat_len,
    )
    for batch_idx in range(len(clip_text_idx_lists)):
        cilp_current_idx = 0
        byt5_current_idx = 0
        for clip_idx, byt5_idx in zip(clip_text_idx_lists[batch_idx], byt5_text_idx_lists[batch_idx]):
            cross_attn_mask[
                batch_idx, 
                cilp_current_idx:cilp_current_idx + len(clip_idx), 
                byt5_current_idx:byt5_current_idx + len(byt5_idx)
            ] = 1
            cilp_current_idx += len(clip_idx)
            byt5_current_idx += len(byt5_idx)

    return dict(
        imgs=imgs,
        pad_masks=pad_masks,
        clip_prompts=pure_clip_prompts,
        text_prompts=text_prompts,
        time_ids=add_time_ids,
        clip_prompt_id_batchs=clip_prompt_id_batchs,
        byt5_text_input_ids=byt5_text_input_ids,
        byt5_attn_mask=byt5_attn_mask,
        glyph_masks=glyph_masks,
        clip_text_feat_idx=clip_text_feat_idx,
        byt5_text_feat_idx=byt5_text_feat_idx,
        refill_idx=refill_idx,
        self_attn_mask=self_attn_mask,
        cross_attn_mask=cross_attn_mask,
    )


def custom_collate_fn_sdxl_long_prompt_canva_render_with_clip_bg_and_text(
    batched_data, 
    tokenizers,
    clip_text_start_spe_token=None,
    clip_text_end_spe_token=None,
    determine_by_tokenizer_idx=0,
    mask_resolution=128,
):
    # imgs are already resized and cropped
    imgs = [item['img'] for item in batched_data]
    text_prompts = [item['text_prompt'] for item in batched_data]
    bg_prompts = [item['bg_prompt'] for item in batched_data]
    valid_prompts = [0 if item['bg_prompt'] == "" else 1 for item in batched_data]
    ori_sizes = [item['ori_size'] for item in batched_data]
    c_crops = [item['c_crop'] for item in batched_data]
    target_sizes = [item['target_size'] for item in batched_data]
    pad_masks = [item['pad_mask'] for item in batched_data]
    glyph_masks = [item['glyph_mask'] for item in batched_data]
    text_bboxes_list = [item['text_bboxes'] for item in batched_data]
    
    # b,c,h,w
    imgs = torch.stack(imgs, dim=0)
    # b,1,h,w
    if pad_masks[0] is not None:
        pad_masks = torch.stack(pad_masks, dim=0).unsqueeze(1)
    else:
        pad_masks = None
    
    # Compute text token id
    # each entry in text_input_id_batchs is a 
    # batched tokenization results of one tokenizer
    bg_prompt_id_batches = []
    for tokenizer in tokenizers:
        pad_token = tokenizer.pad_token_id
        total_tokens = tokenizer(bg_prompts, truncation=False)['input_ids']
        bos = total_tokens[0][0]
        eos = total_tokens[0][-1]
        total_tokens = [i[1:-1] for i in total_tokens]
        new_total_tokens = []
        for token_ids in total_tokens:
            new_total_tokens.append([])
            empty_flag = True
            while len(token_ids) >= 75:
                head_75_tokens = [token_ids.pop(0) for _ in range(75)]
                temp_77_token_ids = [bos] + head_75_tokens + [eos]
                new_total_tokens[-1].append(temp_77_token_ids)
                empty_flag = False
            if len(token_ids) > 0 or empty_flag:
                padding_len = 75 - len(token_ids)
                temp_77_token_ids = [bos] + token_ids + [eos] + [pad_token] * padding_len
                new_total_tokens[-1].append(temp_77_token_ids)
        max_77_len = len(max(new_total_tokens, key=len))
        for new_tokens in new_total_tokens:
            if len(new_tokens) < max_77_len:
                padding_len = max_77_len - len(new_tokens)
                new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b,segment_len,77
        new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
        bg_prompt_id_batches.append(new_total_tokens)
    if bg_prompt_id_batches[0].shape[1] > bg_prompt_id_batches[1].shape[1]:
        tokenizer = tokenizers[1]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = bg_prompt_id_batches[0].shape[1] - bg_prompt_id_batches[1].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(bg_prompt_id_batches[1].shape[0], 1, 1)
        bg_prompt_id_batches[1] = torch.cat((bg_prompt_id_batches[1], padding_part), dim=1)
    elif bg_prompt_id_batches[0].shape[1] < bg_prompt_id_batches[1].shape[1]:
        tokenizer = tokenizers[0]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = bg_prompt_id_batches[1].shape[1] - bg_prompt_id_batches[0].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(bg_prompt_id_batches[0].shape[0], 1, 1)
        bg_prompt_id_batches[0] = torch.cat((bg_prompt_id_batches[0], padding_part), dim=1)        


    # text prompt
    text_prompt_id_batches = [None, None]
    
    # List[List[[idx1, idx2], [idx1, idx2], ...], ...]
    clip_text_idx_lists = []
    
    tokenizer = tokenizers[determine_by_tokenizer_idx]
    clip_text_start_spe_token_id = tokenizer.convert_tokens_to_ids(clip_text_start_spe_token)
    clip_text_end_spe_token_id = tokenizer.convert_tokens_to_ids(clip_text_end_spe_token)
    pad_token = tokenizer.pad_token_id
    total_tokens = tokenizer(text_prompts, truncation=False)['input_ids']
    bos = total_tokens[0][0]
    eos = total_tokens[0][-1]
    total_tokens = [i[1:-1] for i in total_tokens]
    new_total_tokens = []
    for token_ids in total_tokens:
        new_total_tokens.append([])
        clip_text_idx_lists.append([])
        # empty_flag for the case that the token_ids is empty
        empty_flag = True
        while len(token_ids) > 0:
            empty_flag = False
            head_76_tokens = [bos]
            while len(head_76_tokens) < 76 and len(token_ids) > 0:
                next_id = token_ids.pop(0)
                if next_id == clip_text_start_spe_token_id:
                    assert len(clip_text_idx_lists[-1]) == 0 or len(clip_text_idx_lists[-1][-1]) == 2
                    current_idx = (len(new_total_tokens[-1])) * 77 + len(head_76_tokens)
                    clip_text_idx_lists[-1].append([current_idx])
                elif next_id == clip_text_end_spe_token_id:
                    assert len(clip_text_idx_lists[-1][-1]) == 1
                    current_idx = (len(new_total_tokens[-1])) * 77 + len(head_76_tokens)
                    clip_text_idx_lists[-1][-1].append(current_idx)
                else:
                    head_76_tokens.append(next_id)
            if len(head_76_tokens) == 76:
                temp_77_token_ids = head_76_tokens + [eos]
                new_total_tokens[-1].append(temp_77_token_ids)
            elif len(token_ids) == 0:
                padding_len = 76 - len(head_76_tokens)
                temp_77_token_ids = head_76_tokens + [eos] + [pad_token] * padding_len
                new_total_tokens[-1].append(temp_77_token_ids)
        if empty_flag:
            temp_77_token_ids = [bos] + [eos] + [pad_token] * 75
            new_total_tokens[-1].append(temp_77_token_ids)
        
    max_77_len = len(max(new_total_tokens, key=len))
    for new_tokens in new_total_tokens:
        if len(new_tokens) < max_77_len:
            padding_len = max_77_len - len(new_tokens)
            new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
    # b,segment_len,77
    new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
    text_prompt_id_batches[determine_by_tokenizer_idx] = new_total_tokens

    another_tokenizer_idx = 1 - determine_by_tokenizer_idx
    tokenizer = tokenizers[another_tokenizer_idx]
    pure_text_prompts = [
        i.replace(clip_text_start_spe_token, '').replace(clip_text_end_spe_token, '') for i in text_prompts
    ]
    pad_token = tokenizer.pad_token_id
    total_tokens = tokenizer(pure_text_prompts, truncation=False)['input_ids']
    bos = total_tokens[0][0]
    eos = total_tokens[0][-1]
    total_tokens = [i[1:-1] for i in total_tokens]
    new_total_tokens = []
    for token_ids in total_tokens:
        new_total_tokens.append([])
        # empty_flag for the case that the token_ids is empty
        empty_flag = True
        while len(token_ids) >= 75:
            head_75_tokens = [token_ids.pop(0) for _ in range(75)]
            temp_77_token_ids = [bos] + head_75_tokens + [eos]
            new_total_tokens[-1].append(temp_77_token_ids)
            empty_flag = False                    
        if len(token_ids) > 0 or empty_flag:
            padding_len = 75 - len(token_ids)
            temp_77_token_ids = [bos] + token_ids + [eos] + [pad_token] * padding_len
            new_total_tokens[-1].append(temp_77_token_ids)
    max_77_len = len(max(new_total_tokens, key=len))
    for new_tokens in new_total_tokens:
        if len(new_tokens) < max_77_len:
            padding_len = max_77_len - len(new_tokens)
            new_tokens.extend([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
    # b,segment_len,77
    new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long)
    text_prompt_id_batches[another_tokenizer_idx] = new_total_tokens
    
    if text_prompt_id_batches[0].shape[1] > text_prompt_id_batches[1].shape[1]:
        tokenizer = tokenizers[1]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = text_prompt_id_batches[0].shape[1] - text_prompt_id_batches[1].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(text_prompt_id_batches[1].shape[0], 1, 1)
        text_prompt_id_batches[1] = torch.cat((text_prompt_id_batches[1], padding_part), dim=1)
    elif text_prompt_id_batches[0].shape[1] < text_prompt_id_batches[1].shape[1]:
        tokenizer = tokenizers[0]
        pad_token = tokenizer.pad_token_id
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        padding_len = text_prompt_id_batches[1].shape[1] - text_prompt_id_batches[0].shape[1]
        # padding_len, 77
        padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
        # b, padding_len, 77
        padding_part = padding_part.unsqueeze(0).repeat(text_prompt_id_batches[0].shape[0], 1, 1)
        text_prompt_id_batches[0] = torch.cat((text_prompt_id_batches[0], padding_part), dim=1)
    
    glyph_attn_masks = torch.zeros(
        len(bg_prompts),
        mask_resolution,
        mask_resolution,
        text_prompt_id_batches[0].shape[1] * text_prompt_id_batches[0].shape[2],
        dtype=torch.int,
    )
    for batch_idx, (bboxes, clip_idx_segments) in enumerate(
        zip(text_bboxes_list, clip_text_idx_lists)
    ):
        if clip_idx_segments == []:
            continue
        else:
            assert len(bboxes) == len(clip_idx_segments)
        for bbox, clip_idx_segment in zip(bboxes, clip_idx_segments):
            bbox = [int(v / 1645 * mask_resolution + 0.5) for v in bbox]
            bbox[2] = max(bbox[2], 1)
            bbox[3] = max(bbox[3], 1)
            bbox[0: 2] =  np.clip(bbox[0: 2], 0, mask_resolution - 1).tolist()
            bbox[2: 4] = np.clip(bbox[2: 4], 1, mask_resolution).tolist()
            glyph_attn_masks[
                batch_idx,
                bbox[1]: bbox[1] + bbox[3], 
                bbox[0]: bbox[0] + bbox[2], 
                clip_idx_segment[0]: clip_idx_segment[1],
            ]
    bg_attn_masks = (glyph_attn_masks.sum(dim=-1) == 0).to(glyph_attn_masks.dtype)
    
    
    # B, 1, 1
    valid_prompts = torch.tensor(valid_prompts)[:, None, None]

    # Additional embeddings needed for SDXL UNet
    add_time_ids = [list(ori_size + c_crop + target_size) 
                    for ori_size, c_crop, target_size 
                    in zip(ori_sizes, c_crops, target_sizes)]
    # batch_size, 6
    add_time_ids = torch.tensor(add_time_ids)
    
    #b,h,w->b,1,h,w
    glyph_masks = torch.stack(glyph_masks, dim=0).unsqueeze(1)
    
    
    return dict(
        imgs=imgs,
        pad_masks=pad_masks,
        bg_prompts=bg_prompts,
        text_prompts=pure_text_prompts,
        time_ids=add_time_ids,
        bg_prompt_id_batches=bg_prompt_id_batches,
        text_prompt_id_batches=text_prompt_id_batches,
        valid_prompts=valid_prompts,
        glyph_masks=glyph_masks,
        glyph_attn_masks=glyph_attn_masks,
        bg_attn_masks=bg_attn_masks,
    )


def custom_collate_fn_pixart_long_prompt_canva_render(
    batched_data, 
    tokenizer,
    bg_prompt_max_length=512,
    byt5_tokenizer=None,
    byt5_max_length=256):    
    # imgs are already resized and cropped
    imgs = [item['img'] for item in batched_data]
    text_prompts = [item['text_prompt'] for item in batched_data]
    bg_prompts = [item['bg_prompt'] for item in batched_data]
    valid_bg_prompts = [0 if item['bg_prompt'] == "" else 1 for item in batched_data]
    pad_masks = [item['pad_mask'] for item in batched_data]
    if 'glyph_attn_mask' in batched_data[0] and batched_data[0]['glyph_attn_mask'] is not None:
        glyph_attn_masks = [item['glyph_attn_mask'] for item in batched_data]
        bg_attn_masks = [torch.sum(item['glyph_attn_mask'], dim=-1) == 0 for item in batched_data]
        glyph_attn_masks = torch.stack(glyph_attn_masks, dim=0)
        bg_attn_masks = torch.stack(bg_attn_masks, dim=0).to(glyph_attn_masks.dtype)
    else:
        glyph_attn_masks = None
        bg_attn_masks = None
    
    # b,c,h,w
    imgs = torch.stack(imgs, dim=0)
    # b,1,h,w
    if pad_masks[0] is not None:
        pad_masks = torch.stack(pad_masks, dim=0).unsqueeze(1)
    else:
        pad_masks = None
    
    # Compute text token id
    # each entry in text_input_id_batchs is a 
    # batched tokenization results of one tokenizer
    
    bg_prompt_inputs = tokenizer(
        bg_prompts,
        padding="max_length",
        max_length=bg_prompt_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    bg_prompt_id_batchs = bg_prompt_inputs.input_ids

    bg_prompt_attn_mask = bg_prompt_inputs.attention_mask
    bg_prompt_attn_mask= bg_prompt_attn_mask>0
    bg_prompt_attn_mask= bg_prompt_attn_mask.squeeze(0)

    byt5_text_inputs = byt5_tokenizer(
        text_prompts,
        padding="max_length",
        max_length=byt5_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    # b, byt5_max_length
    byt5_text_input_ids = byt5_text_inputs.input_ids
    # b, byt5_max_length
    byt5_attn_mask = byt5_text_inputs.attention_mask
    
    # B, 1, 1
    valid_bg_prompts = torch.tensor(valid_bg_prompts)[:, None, None]

    #merge masks
    bg_prompt_attn_mask_inflate = bg_prompt_attn_mask.unsqueeze(1).unsqueeze(1).repeat(1,bg_attn_masks.shape[1],bg_attn_masks.shape[2],1)
    bg_attn_masks=bg_attn_masks.unsqueeze(3).repeat(1,1,1,bg_prompt_attn_mask_inflate.shape[3]).to(torch.int) & bg_prompt_attn_mask_inflate.to(torch.int)

    glyph_attn_masks = byt5_attn_mask.unsqueeze(1).unsqueeze(1).repeat(1,glyph_attn_masks.shape[1],glyph_attn_masks.shape[2],1).to(torch.int) & glyph_attn_masks.to(torch.int)
    #全变成[64,64,64,512]



        
    return dict(
        imgs=imgs,  #[64,3,512,512] 隐空间w和h为64
        pad_masks=pad_masks, #None
        bg_prompts=bg_prompts,
        text_prompts=text_prompts,
        bg_prompt_id_batchs=bg_prompt_id_batchs, #[64,512]
        valid_bg_prompts=valid_bg_prompts,
        byt5_text_input_ids=byt5_text_input_ids, #[64,512]
        glyph_attn_masks=glyph_attn_masks, #[64,64,64,512]
        bg_attn_masks=bg_attn_masks,#[64,64,64]
        byt5_attn_mask=byt5_attn_mask,#[64,512]
        bg_prompt_attn_mask=bg_prompt_attn_mask,#[64,512]
        
    )