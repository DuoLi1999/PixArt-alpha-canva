from diffusers import ModelMixin
import torch.nn as nn
import torch
class ByT5Mapper(ModelMixin):
    def __init__(self, byt5_output_dim=1472, pixart_hidden_dim=1152, uncond_prob=0.1, token_num=512):
        super().__init__()
        self.mapper = nn.Sequential(
                nn.LayerNorm(byt5_output_dim),
                nn.Linear(byt5_output_dim, pixart_hidden_dim),
                nn.ReLU(),
                nn.Linear(pixart_hidden_dim, pixart_hidden_dim)
        )
        self.register_buffer("y_embedding_1", nn.Parameter(torch.randn(token_num, byt5_output_dim) / byt5_output_dim ** 0.5))
        self.uncond_prob = uncond_prob



    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding_1, caption)
        return caption

    def forward(self, byt5_embedding, train, force_drop_ids=None):
        if train:
            assert byt5_embedding.shape[2:] == self.y_embedding_1.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            byt5_embedding = self.token_drop(byt5_embedding, force_drop_ids)
        byt5_embedding = self.mapper(byt5_embedding)
        return byt5_embedding
    