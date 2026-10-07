# The backbones compared in the separability study. Each loader returns a frozen model
# that maps a (B,3,224,224) float batch in [0,1] to a (B,D) embedding, plus the
# normalisation the model was trained with, so extract.py can share one image read across
# all of them.

import torch

from models_features_extraction.builder import vit_small

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _vits_dino():
    return vit_small(pretrained=True, progress=False, key='DINO_p16', patch_size=16), IMAGENET, 384


def _resnet50():
    import timm
    return timm.create_model('resnet50.tv_in1k', pretrained=True, num_classes=0), IMAGENET, 2048


def _phikon_v2():
    from transformers import AutoModel
    m = AutoModel.from_pretrained('owkin/phikon-v2')

    class Cls(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).last_hidden_state[:, 0]

    return Cls(m), IMAGENET, 1024


def _midnight():
    # Kaiko Midnight-12k (ViT-g); its model card embeds as [CLS ; mean of patch tokens]
    from transformers import AutoModel
    m = AutoModel.from_pretrained('kaiko-ai/midnight')

    class ClsMean(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            h = self.m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)

    return ClsMean(m), HALF, 3072


LOADERS = {
    'vits_dino': _vits_dino,
    'resnet50_in': _resnet50,
    'phikon_v2': _phikon_v2,
    'midnight': _midnight,
}


class Encoder(torch.nn.Module):
    """Normalises a [0,1] batch with the backbone's own statistics and embeds it."""

    def __init__(self, name, device):
        super().__init__()
        model, (mean, std), self.dim = LOADERS[name]()
        self.model = model.eval().to(device)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.model((x - self.mean) / self.std).float()
