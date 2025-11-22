import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class ViT(nn.Module):
    def __init__(self, num_classes=10, use_pretrained=False):
        weights = ViT_B_16_Weights.DEFAULT if use_pretrained else None  # set to None to avoid downloads
        super().__init__()
        self.vit = vit_b_16(weights=weights)

        # replace classifier head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

        # make patch embeddings contiguous after transpose
        def _process_input(x):
            x = self.vit.conv_proj(x)                 # (N, embed_dim, H/16, W/16)
            x = x.flatten(2).transpose(1, 2).contiguous()  # (N, num_patches, embed_dim)
            return x
        self.vit._process_input = _process_input

    def forward(self, x):
        return self.vit(x)

