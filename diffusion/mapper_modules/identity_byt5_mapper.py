from diffusers import ModelMixin

class IdentityByT5Mapper(ModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
            
    def forward(self, inputs_embeds, attention_mask):
        return inputs_embeds
