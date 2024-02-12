from diffusion.data.builder import build_dataset, build_dataloader, set_data_root  
from canva.dataset import Canva8ChannelsDataset    
if __name__ == '__main__':
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
            )   
    train_dataloader = build_dataloader(dataset, num_workers=8, batch_size=200, shuffle=True)
