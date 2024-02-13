from diffusion.data.builder import build_dataset, build_dataloader, set_data_root  
import torch
from train_scripts.canva.sample_dataset import Canva8ChannelsDataset    
if __name__ == '__main__':
    blob_path='/pyy/openseg_blob'
    dataset = Canva8ChannelsDataset(
            resolution=512,
            proportion_empty_prompts=0.0,
            use_embed=True,
            prompt_with_text=True,
            img_path= blob_path +'/weicong/big_file/data/canva-data/canva-render-10.19/',
            ann_path= blob_path +'/weicong/big_file/data/canva-data/canva-binmask/canva-binmask-index.json',
            embed_dir= blob_path +'/weicong/big_file/data/canva-data/canva-binmask-t5-a100/',
            category='all',
            img_type='img', # all(8), aux(8+3), img(3), bg(3)
            target_img_type=None,
            prompt_mode = 'whole',
            )   
    train_dataloader = build_dataloader(dataset, num_workers=0, batch_size=200, shuffle=True)
    for data in train_dataloader:
        with open('samples/text_embed.pt', 'wb') as f:
                torch.save(data[1],f)
        with open('samples/mask.pt', 'wb') as f:
                torch.save(data[2], f)
        break
#     dataset.info('samples/sample_info.json')
