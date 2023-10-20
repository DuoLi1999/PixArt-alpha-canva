import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from scripts.interface import mask_feature
from diffusion import IDDPM, DPMS
from scripts.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, get_chunks


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--t5_path', default='output/pretrained_models/t5_ckpts', type=str)
    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--txt_file', default='asset/samples.txt', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.0, type=float)
    parser.add_argument('--sampling_algo', default='dpms', type=str, choices=['iddpm', 'dpms'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)

    return parser.parse_args()


def set_env(seed):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale):

    hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
    ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = chunk
        null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        with torch.no_grad():
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:, None]
            masked_embs, keep_index = mask_feature(caption_embs, emb_masks)
            print(f'finish embedding')

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)

                model_kwargs = dict(y=torch.cat([masked_embs, null_y[:, :, :keep_index, :]]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar})
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpms':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar})
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=masked_embs,
                                  uncondition=null_y[:, :, :keep_index, :],
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )

        samples = vae.decode(samples / 0.18215).sample
        torch.cuda.empty_cache()
        # Save and display images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpms']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    lewei_scale = {512: 1, 1024: 2}     # trick for pos embedding adoption
    sample_steps_dict = {'iddpm': 100, 'dpms': 20}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]

    # model setting
    if args.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)

    print("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    work_dir = os.path.join(*args.model_path.split('/')[:-2])
    work_dir = '/'+work_dir if args.model_path[0] == '/' else work_dir

    # data setting
    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*.pth', args.model_path).group(1)
    except:
        epoch_name = '-unknown'
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale)