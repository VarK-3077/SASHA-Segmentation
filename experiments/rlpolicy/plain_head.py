# The plain per-token head for the "ViT alone" ablation: the level-2 token head with everything but a per-token
# two-layer MLP switched off. It is tokhead's TokenHead with the 15 x 15 context branch and the crop-mean branch
# fixed at zero (weights and biases zeroed and frozen), so each token's logit depends on that token only:
#     logit = cls(act(proj(token))),   proj: 1024 -> 256,   cls: 256 -> 1
# act is the head's own pointwise activation path with the two context terms equal to zero, i.e. GELU applied
# three times. Keeping the TokenHead class means the cache trainer, the PCA folding (tokcache.export), tokhead's
# inference and the cascade runners load it unchanged.
# Train: python -m experiments.tokcache.train ... --head experiments.rlpolicy.plain_head:PlainHead

from experiments.tokhead.model import TokenHead


class PlainHead(TokenHead):
    def __init__(self, **kw):
        super().__init__(**kw)
        frozen = list(self.ctx.parameters()) + (list(self.glob.parameters()) if hasattr(self, 'glob') else [])
        for p in frozen:
            p.data.zero_()
            p.requires_grad_(False)
