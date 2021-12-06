import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1, keepdim = True)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# transformer

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm_out(x)

# text and vision transformers

class TextTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(dim, **kwargs)

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask)
        return out

class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        patch_size,
        channels,
        **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.transformer = Transformer(dim, **kwargs)

    def forward(self, x):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        out = self.transformer(x)
        return out

# main clip class

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        num_visual_tokens = 512,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_image_size = 256,
        visual_patch_size = 32,
        channels = 3,
        use_all_token_embeds = False
    ):
        super().__init__()
        self.text_transformer = TextTransformer(
            dim = dim_text,
            num_tokens = num_text_tokens,
            max_seq_len = text_seq_len,
            depth = text_enc_depth,
            heads = text_heads
        )

        self.visual_transformer = VisionTransformer(
            dim = dim_image,
            image_size = visual_image_size,
            patch_size = visual_patch_size,
            channels = channels,
            depth = visual_enc_depth,
            heads = visual_heads
        )

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

    def forward(
        self,
        text,
        image,
        text_mask = None,
        return_loss = False
    ):
        b, device = text.shape[0], text.device

        enc_text = self.text_transformer(text, mask = text_mask)
        enc_image = self.visual_transformer(image)

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        if self.use_all_token_embeds:
            text_embeds = enc_text[:, 1:]
            image_embeds = enc_image[:, 1:]
        else:
            text_embeds = enc_text[:, 0]
            image_embeds = enc_image[:, 0]

        # project to latents

        text_latents = self.to_text_latent(text_embeds)
        image_latents = self.to_visual_latent(image_embeds)
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # get temperature

        temp = self.temperature.exp()

        if not return_loss and self.use_all_token_embeds:
            return einsum('b t d, b i d -> b t i', text_latents, image_latents) * temp

        if not return_loss and not self.use_all_token_embeds:
            return einsum('b d, b d -> b', text_latents, image_latents) * temp

        # contrastive loss

        if self.use_all_token_embeds:
            # fine-grained CLIP logic
            sim = einsum('x t d, y i d -> x y t i', text_latents, image_latents) * temp
            text_to_image = reduce(reduce(sim, 'bt bi t i -> bt bi t', 'max'), 'bt bi t -> bt bi', 'mean')
            image_to_text = reduce(reduce(sim, 'bt bi t i -> bt bi i', 'max'), 'bt bi i -> bt bi', 'mean')
        else:
            sim = einsum('t d, i d -> t i', text_latents, image_latents) * temp
            text_to_image, image_to_text = sim, sim.t()

        labels = torch.arange(b, device = sim.device)
        loss = (F.cross_entropy(text_to_image, labels) + F.cross_entropy(image_to_text, labels)) / 2
        return loss
