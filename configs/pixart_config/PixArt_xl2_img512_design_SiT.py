_base_ = ['../PixArt_xl2_design.py']
data = dict(type='canva', load_vae_feat=False)

image_size = 512

# model setting
window_block_indexes=[]
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2'
fp32_attention = False #True
load_from = False
vae_pretrained = "/pyy/yuyang_blob/pyy/code/PixArt-alpha-canva/output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0
model_max_length=512
use_flash_attn= False

# training setting
use_fsdp= False   # if use FSDP mode
num_workers=4
train_batch_size = 64 # 32
num_epochs = 200 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)
noise_offset=0.0
zero_snr = False
mixed_precision = 'bf16'

eval_sampling_steps = 200
log_interval = 20
save_model_epochs=1
work_dir = 'output/SiT_base'


