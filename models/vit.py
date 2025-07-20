import torch
import torch.nn as nn
from einops import rearrange, repeat
from config.config import config

class ViT(nn.Module):
    def __init__(self, num_classes=config['num_classes']):
        super().__init__()
        self.patch_size = config['vit_patch_size']
        self.dim = config['vit_dim']

        self.patch_embed = nn.Conv2d(3,self.dim,kernel_size=self.patch_size,stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.rand(1, (config['image_size'] // self.patch_size) ** 2 + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=config['vit_heads'],
                dim_feedforward=config['vit_mlp_dim'],
                activation='gelu'
            ) for _ in range(config['vit_depth'])
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, num_classes)
        )
        self.norm = nn.LayerNorm(self.dim)

        
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        return self.mlp_head(x[:, 0])