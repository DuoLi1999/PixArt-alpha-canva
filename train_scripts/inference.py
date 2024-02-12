import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning

from tqdm import tqdm
import torch

import torchvision.utils
from diffusion import DPMS
import os

from diffusion.data.datasets import get_chunks, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST



@torch.inference_mode()
def visualize(model,vae,path,step,device,name=None,bs=1, resolution=512, ar=1.,sample_steps=20, cfg_scale=4.5):
    if not os.path.exists(f'samples/{name}'):
        os.mkdir(f'samples/{name}')

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
            model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=None)#改掉mask
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
    torchvision.utils.save_image(samples_tensor, f'samples/{name}/image_{step}.png')





