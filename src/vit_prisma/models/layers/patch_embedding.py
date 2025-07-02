import logging
import torch.nn as nn
import torch 
from jaxtyping import Float
from vit_prisma.configs import HookedViTConfig
import einops

class PatchEmbedding(nn.Module):

    def __init__(self, config, logger=None):
        super().__init__()
        self.logger = logger
        self.config = config
        self.proj = nn.Conv2d(
            self.config.n_channels, 
            self.config.d_model, 
            kernel_size=self.config.patch_size, 
            stride=self.config.patch_size, 
            bias=True
        )
    
    def _log(self, stage, tensor):
        if self.logger:
            self.logger.info(f"{stage} size: {tensor.shape}")

    def forward(self, x:Float[torch.Tensor, "batch channel height width"]) -> Float[torch.Tensor, "batch n_tokens d_model"]:
        self._log("PatchEmbedding input", x)

        x = self.proj(x).flatten(2).transpose(1, 2)

        self._log("PatchEmbedding output", x)
        return x
    

# Used for videos, 3 dimensional spacetime patches 
class TubeletEmbedding(nn.Module):

    def __init__(self, cfg:HookedViTConfig):
        super().__init__()
        self.cfg = cfg


        print(cfg)
        tubelet_size = [self.cfg.tubelet_size, self.cfg.patch_size, self.cfg.patch_size]
        self.proj = nn.Conv3d(
            self.cfg.n_channels, 
            self.cfg.d_model, 
            kernel_size=tubelet_size, 
            stride=tubelet_size, 
            bias=True
        )

    def forward(self, x:Float[torch.Tensor, "batch num_frames channels height width"]) -> Float[torch.Tensor, "batch n_tokens d_model"]:
        
        # Flip num_frames and channels
        x = einops.rearrange(x, "b t c h w -> b c t h w")
        print("X SHAPE", x.shape)
        
        x = self.proj(x)

        # Flatten the tokens
        x = einops.rearrange(x, "b c t h w -> b (t h w) c") 

        return x

