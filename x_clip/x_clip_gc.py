import os
import math
import copy
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torch.distributed as dist
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_clip.mlm import MLM
from x_clip.visual_ssl import SimSiam, SimCLR

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

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
        image_encoder = None,
        text_encoder = None,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        text_has_cls_token = True,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_has_cls_token = True,
        channels = 3,
        use_all_token_embeds = False,
        downsample_image_embeds = False,
        decoupled_contrastive_learning = False,
        extra_latent_projection = False,
        use_mlm = False,
        text_ssl_loss_weight = 0.05,
        use_visual_ssl = False,
        visual_ssl_type = 'simsiam',
        visual_ssl_hidden_layer = -1,
        simclr_temperature = 0.1,
        image_ssl_loss_weight = 0.05,
        loss_over_ranks = False,
        rank = None,
        grad_cache = False,
    ):
        super().__init__()
        assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'

        # instantiate text transformer

        self.text_has_cls_token = text_has_cls_token

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens = num_text_tokens + (1 if use_mlm else 0),
                max_seq_len = text_seq_len,
                depth = text_enc_depth,
                heads = text_heads
            )

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size = visual_image_size,
                patch_size = visual_patch_size,
                channels = channels,
                depth = visual_enc_depth,
                heads = visual_heads
            )

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight

        if use_mlm:
            self.mlm = MLM(
                self.text_transformer,
                dim = dim_text,
                num_tokens = num_text_tokens
            )

        # image ssl

        self.use_visual_ssl = use_visual_ssl
        self.image_ssl_loss_weight = image_ssl_loss_weight

        if use_visual_ssl:
            if visual_ssl_type == 'simsiam':
                ssl_type = SimSiam
            elif visual_ssl_type == 'simclr':
                ssl_type = partial(SimCLR, temperature = simclr_temperature)
            else:
                raise ValueError(f'unknown visual_ssl_type')

            self.visual_ssl = ssl_type(
                self.visual_transformer,
                image_size = visual_image_size,
                hidden_layer = visual_ssl_hidden_layer
            )

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # image latent projection

        if downsample_image_embeds:
            assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'

            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv2d(dim_image, dim_image, 4, stride = 2, padding = 1, bias = False, groups = dim_image),
                nn.Conv2d(dim_image, dim_latent, 1),
                Rearrange('b c h w -> b (h w) c')
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # proposed in https://arxiv.org/abs/2110.11316 (CLOOB)
        self.extra_latent_projection = extra_latent_projection

        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        # loss ove ranks
        self.loss_over_ranks = loss_over_ranks
        self.rank = rank
        self.grad_cache = grad_cache


    def forward_text(
        self,
        text,
        text_mask = None,
        freeze_text_encoder = False,    # text encoder is not trained if this is set to True
    ):
        b, device = text.shape[0], text.device

        # get encoded text

        text_encoding_context = null_context if not freeze_text_encoder else torch.no_grad

        with text_encoding_context():
            enc_text = self.text_transformer(text, mask = text_mask)

            if freeze_text_encoder:
                enc_text.detach_()

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        if self.use_all_token_embeds:
            text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
        else:
            text_embeds = enc_text[:, 0]

        # project to latents

        text_latents = self.to_text_latent(text_embeds)
        text_latents = l2norm(text_latents)

        return text_latents


    def forward_image(
        self,
        image,
        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
    ):
        b, device = text.shape[0], text.device

        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT

        image_encoding_context = null_context if not freeze_image_encoder else torch.no_grad

        with image_encoding_context():
            enc_image = self.visual_transformer(image)

            if freeze_image_encoder:
                enc_image.detach_()

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        if self.use_all_token_embeds:
            image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
        else:
            image_embeds = enc_image[:, 0]

        # project to latents

        image_latents = self.to_visual_latent(image_embeds)
        image_latents = l2norm(image_latents)

        # early return, if needed

        return image_latents


# TO DO:
# 1.) Add SSL for both modalities in a extra methods.
# 2.) Add ...latents_extra setup
# 3.) Add early return of similarity matrix again


# ALL-IN-ONE FORWARD TEMPLATE
#    def forward(
#        self,
#        text,
#        image,
#        text_mask = None,
#        return_loss = True,
#        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
#        freeze_text_encoder = False,    # text encoder is not trained if this is set to True
#        text_to_image = True            # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
#    ):
#        b, device = text.shape[0], text.device
#
#        # ssl
#
#        text_ssl_loss = 0
#        image_ssl_loss = 0
#
#        if return_loss:
#            text_ssl_loss = self.mlm(text, mask = text_mask) if self.use_mlm else 0
#            image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0
#
#        # get encoded text
#
#        text_encoding_context = null_context if not freeze_text_encoder else torch.no_grad
#
#        with text_encoding_context():
#            enc_text = self.text_transformer(text, mask = text_mask)
#
#            if freeze_text_encoder:
#                enc_text.detach_()
#
#        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT
#
#        image_encoding_context = null_context if not freeze_image_encoder else torch.no_grad
#
#        with image_encoding_context():
#            enc_image = self.visual_transformer(image)
#
#            if freeze_image_encoder:
#                enc_image.detach_()
#
#        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only
#
#        if self.use_all_token_embeds:
#            text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
#            image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
#        else:
#            text_embeds = enc_text[:, 0]
#            image_embeds = enc_image[:, 0]
#
#        # project to latents
#
#        text_latents = self.to_text_latent(text_embeds)
#        image_latents = self.to_visual_latent(image_embeds)
#        text_latents, image_latents = map(l2norm, (text_latents, image_latents))
#
#        # calculate another set of latents for image to text (vs text to image)
#        # proposed by CLOOB
#
#        text_latents_extra, image_latents_extra = text_latents, image_latents
#        if self.extra_latent_projection:
#            text_latents_extra = self.to_text_latent_extra(text_embeds)
#            image_latents_extra = self.to_visual_latent_extra(image_embeds)
#            text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))
#
#        # get temperature
#
#        temp = self.temperature.exp()
#
#        # early return, if needed
#
#        if not return_loss and self.use_all_token_embeds:
#            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
#            return einsum('b t d, b i d -> b t i', *einsum_args) * temp
#
#        if not return_loss and not self.use_all_token_embeds:
#            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
#            return einsum('b d, b d -> b', *einsum_args) * temp
#
#        # TO DO: Add _extra outputs for returns below
#
#        if self.grad_cache:
#            return torch.stack((text_latents, image_latents), dim=0)
#        
#        return text_latents, image_latents


    def loss(self, text_latents, image_latents):
        # TO DO: Check if there is a better way than to recalculate b and temp from model_forward?
        b, device = text_latents.shape[0], text_latents.device
        temp = self.temperature.exp()
        
        # contrastive loss

        if self.loss_over_ranks:
            all_text_latents  = all_gather(text_latents)
            all_image_latents = all_gather(image_latents)

        if self.use_all_token_embeds:
            # fine-grained CLIP logic
            if self.loss_over_ranks:
                sim_text_to_image = einsum('x t d, y i d -> x y t i', text_latents, all_image_latents) * temp
                sim_image_to_text = einsum('y i d, x t d -> y x i t', image_latents, all_text_latents) * temp
            else:
                sim_text_to_image = einsum('x t d, y i d -> x y t i', text_latents, image_latents) * temp
                sim_image_to_text = sim_text_to_image # TO DO: With .permute(1,0,3,2) we could skip the next two else below,
                # this would save some lines of code but is maybe slower?

            # TO DO: extra_latent_projection is not adapted with loss_over_ranks, needs to be fixed!
            if self.extra_latent_projection:
                sim_image_to_text = einsum('x t d, y i d -> x y t i', text_latents_extra, image_latents_extra) * temp

            if exists(text_mask):
                text_to_image = reduce(sim_text_to_image, 'bt bi t i -> bt bi t', 'max')
                text_to_image_mask = rearrange(text_mask, 'bt t -> bt 1 t')
                text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

                if self.loss_over_ranks:
                    image_to_text_mask = rearrange(text_mask, 'bt t -> 1 bt 1 t')
                    masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
                    image_to_text = reduce(reduce(masked_sim, 'bi bt i t -> bi bt i', 'max'), 'bi bt i -> bi bt', 'mean')
                else: # This else part can be removed, see comment above.
                    image_to_text_mask = rearrange(text_mask, 'bt t -> bt 1 t 1')
                    masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
                    image_to_text = reduce(reduce(masked_sim, 'bt bi t i -> bt bi i', 'max'), 'bt bi i -> bi bt', 'mean')
            else:
                text_to_image = reduce(reduce(sim_text_to_image, 'x y t i -> x y t', 'max'), 'x y t -> x y', 'mean')
                if self.loss_over_ranks:
                    image_to_text = reduce(reduce(sim_image_to_text, 'y x i t -> y x i', 'max'), 'y x i -> y x', 'mean')
                else: # This else part can be removed, see comment above.
                    image_to_text = reduce(reduce(sim_image_to_text, 'x y t i -> x y i', 'max'), 'x y i -> y x', 'mean')
        else:
            if self.loss_over_ranks:
                text_to_image = einsum('x d, y d -> x y', text_latents, all_image_latents) * temp
                image_to_text = einsum('y d, x d -> y x', image_latents, all_text_latents) * temp
            else:
                text_to_image = einsum('x d, y d -> x y', text_latents, image_latents) * temp
                image_to_text = text_to_image.t()

            # TO DO: extra_latent_projection does not work with loss_over_ranks, needs to be fixed!
            if self.extra_latent_projection:
                image_to_text = einsum('x d, y d -> y x', text_latents_extra, image_latents_extra) * temp

        # calculate loss

        # exponentiate

        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # numerators
        if self.loss_over_ranks:
            pos_diag = self.rank * b # the positive diagonal for the rank is at position rank * b
        else:
            pos_diag = 0

        text_to_image_pos, image_to_text_pos = map(partial(torch.diag, diagonal=pos_diag), (text_to_image_exp, image_to_text_exp))

        # denominator

        if self.decoupled_contrastive_learning:
            if self.loss_over_ranks:
                # TO DO: This needs some testing!
                pos_mask = torch.zeros_like(text_to_image_exp, device = device, dtype = torch.bool)
                pos_mask[torch.arange(b), torch.arange(b*self.rank, b*(self.rank+1))] = True

                # use all latents
                # TO DO: Needs to be setup for all latents.
            else:
                pos_mask = torch.eye(b, device = device, dtype = torch.bool)

            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

        # loss

        # TO DO: Change the calculation to log space, like in:
        # https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L89
        text_to_image_loss = -torch.log(text_to_image_pos / text_to_image_denom).mean()
        image_to_text_loss = -torch.log(image_to_text_pos / image_to_text_denom).mean()

        cl_loss = (text_to_image_loss + image_to_text_loss) / 2

        return cl_loss
#        # TO DO: Check loss difference due to re-weighting below!
#
#        # calculate weights
#
#        cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight)
#
#        loss = (cl_loss * cl_loss_weight) \
#            + (text_ssl_loss * self.text_ssl_loss_weight) \
#            + (image_ssl_loss * self.image_ssl_loss_weight)
#
#        return loss


    def forward_with_loss(
        self,
        text,
        image,
        text_mask = None,
        freeze_text_encoder = False,
        freeze_image_encoder = False,
        ):

#        text_latents, image_latents = self.forward(
#            text,
#            image,
#            text_mask = None,
#            **kwargs
#            )

        text_latents = self.forward_text(text, text_mask, freeze_text_encoder)
        image_latents = self.forward_image(image, freeze_image_encoder)

        loss = self.loss(text_latents, image_latents)

        return loss


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_start = torch.distributed.get_rank() * ctx.batch_size
        idx_end = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_start:idx_end]

all_gather = AllGather.apply
