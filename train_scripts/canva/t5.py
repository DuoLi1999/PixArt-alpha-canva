from diffusers import DiffusionPipeline
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import huggingface_hub
from tqdm import tqdm

@torch.no_grad()
def encode_prompt(
    prompt,
    tokenizer,
    text_encoder,
    max_length = 512,
):

    # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
    # max_length = 512

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    prompt_embeds = text_encoder(
        text_input_ids.to(text_encoder.device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds