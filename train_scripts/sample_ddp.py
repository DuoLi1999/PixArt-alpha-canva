# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import torch
import torch.distributed as dist
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
import torchvision.utils
from diffusion import DPMS
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.model.builder import build_model




def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

@torch.inference_mode()
def visualize(model,vae,path,step,device,bs=1, resolution=512, ar=1.,sample_steps=20, cfg_scale=4.5):
    caption_embs = torch.load(path, map_location=torch.device(device))
    model.eval()
    latent_size = resolution // 8

    samples_list=[]
    for chunk in tqdm(list(get_chunks(caption_embs, bs)), unit='batch'):
        # if bs == 1:     
        hw = torch.tensor([[resolution,resolution]], dtype=torch.float, device=device).repeat(bs, 1)
        ar = torch.tensor([[ar]], device=device).repeat(bs, 1)
        latent_size_h, latent_size_w = latent_size, latent_size

        null_y = model.y_embedder.y_embedding_1[None].repeat(1, 1, 1)[:, None]

        with torch.no_grad():
            caption_embs = chunk[0].float()[None,None]
            # print('finish embedding')

            # Create sampling noise:
            z = torch.randn(1, 4, latent_size_h, latent_size_w, device=device)
            model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=None)
            dpm_solver = DPMS(model.forward_with_dpmsolver,
                                condition=caption_embs,
                                uncondition=null_y,
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
        samples_list.append(samples.squeeze(0))
        torch.cuda.empty_cache()
    samples_tensor = torch.stack(samples_list)
    torchvision.utils.save_image(samples_tensor, f'output/image_{step}.png')



def main(args):
    """
    Run sampling.
    """
    config = read_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    assert args.num_fid_samples==50_000, "num_fid_samples must be 50_000"
    

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Load model:
    latent_size = args.resolution // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}

    # build models
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()

    # load model from checkpoint
    ckpt_path = args.ckpt 
    load_checkpoint(ckpt_path, model, load_ema=False)
    model.to(device).eval()  # important!
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"

    folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                  f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                  f"{args.sample_steps}"
   
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    
    from canva.dataset import Canva8ChannelsDataset   
    dataset = Canva8ChannelsDataset(
            resolution=512,
            proportion_empty_prompts=0.0,
            use_embed=True,
            prompt_with_text=True,
            img_path= args.blob_path +'/weicong/big_file/data/canva-data/canva-render-10.19/',
            ann_path= args.blob_path +'/weicong/big_file/data/canva-data/canva-binmask/canva-binmask-index.json',
            embed_dir= args.blob_path +'/weicong/big_file/data/canva-data/canva-binmask-t5-a100/',
            category='all',
            img_type='img', # all(8), aux(8+3), img(3), bg(3)
            target_img_type=None,
            prompt_mode = 'whole',
            train='val'
            )   
    
    for i in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        
        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        dist.barrier()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,required=True,
                        help="path to a Pixart checkpoint.")
    parser.add_argument("--config", type=str, default='configs/pixart_config/PixArt_xl2_img512_design.py',
                        help="config_path")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--resolution", type=int, choices=[256, 512,1024], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=4.5)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--aspect_ratio", type=float, default=1.)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument('--blob-path', default='/pyy/openseg_blob' )
    

    args = parser.parse_known_args()[0]
    main(args)
