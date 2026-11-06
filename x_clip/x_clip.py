import math
import copy
from collections import namedtuple
from contextlib import nullcontext
from functools import partial, wraps
from itertools import combinations

import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from torch_einops_utils import masked_mean
from torch_einops_utils.math import l2norm
from torch_einops_utils.shape import assert_shape, shape, size
from torch_einops_utils.torch_einops_utils import default, exists, identity, maybe

from x_clip.mlm import MLM
from x_clip.visual_ssl import SimSiam, SimCLR
from x_clip.distributed import all_gather

# helper functions

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

maybe_rearrange = maybe(rearrange)

def mask_fill(t, mask, value = None):
    if not exists(mask):
        return t

    value = default(value, max_neg_value(t.dtype))
    return t.masked_fill(~mask, value)

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def matrix_diag(t):
    device = t.device
    i, j = shape(t, '... [i] [j]')
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

def off_diagonal(t):
    i, j = shape(t, '... [i] [j]')
    assert i == j, 'off diagonal requires square matrices'
    mask = ~torch.eye(i, device = t.device, dtype = torch.bool)
    return t[..., mask]

def off_diagonal_mse(t):
    t = off_diagonal(t)
    return (t ** 2).mean()

def spectral_contrastive_loss(latents_a, latents_b):
    sim = einsum('m i d, n j d -> m n i j', latents_a, latents_b)
    return -2 * matrix_diag(sim).mean() + off_diagonal_mse(sim)

def logsumexp_regularizer(sim):
    return (torch.logsumexp(sim, dim = -1) ** 2).mean()

def average_directions(a_to_b, b_to_a, asymmetric = False):
    return a_to_b if asymmetric else (a_to_b + b_to_a) / 2

def infonce_from_sim(sim, temp = 1., decoupled = False):
    sim = sim * temp
    pos = matrix_diag(sim.exp())

    exp = sim.exp()
    if decoupled:
        exp = exp.masked_fill(torch.eye(size(sim, '... [j]'), device = sim.device, dtype = torch.bool), 0.)

    return (-log(pos) + log(exp.sum(dim = -1))).mean(dim = -1)

# losses

Losses = namedtuple('Losses', [
    'loss',
    'contrastive',
    'multiview',
    'sim_reg',
    'spectral',
    'logsumexp',
    'negative_mse',
    'ssl'
])

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = {key[len(prefix):]: value for key, value in kwargs_with_prefix.items()}
    return kwargs_without_prefix, kwargs

# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        num_patches = size(x, 'b [n] c')
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(num_patches)))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n = shape(x, '[b] [n] ...')
        device = x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = size(freqs, '... [d]')
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            nn.LayerNorm(inner_dim, bias = False),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), nn.LayerNorm(dim, bias = False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = shape(sim, '... [i] [j]')
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

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
        causal = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        checkpoint_during_training = False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_in = nn.LayerNorm(dim, bias = False)
        self.norm_out = nn.LayerNorm(dim, bias = False)

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        mask = None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
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
        dim_head,
        rotary_pos_emb = None,
        causal = False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    def forward(self, x, mask = None):
        b, n = shape(x, '[b] [n]')
        device = x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            seq_len = n + (1 if exists(self.cls_token) else 0)
            rotary_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        patch_size,
        channels,
        patch_dropout = 0.5,
        **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(
        self,
        x,
        keep_all_patches = False
    ):
        device, is_video = x.device, x.ndim == 5

        if is_video:
            b, t = shape(x, '[b] [t] ...')
            x = rearrange(x, 'b t ... -> (b t) ...')

        x = self.to_tokens(x)
        n = size(x, 'b [n] d')

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)

        if is_video:
            out = rearrange(out, '(b t) n d -> b (t n) d', b = b, t = t)
            cls_tokens = rearrange(cls_tokens, '(b t) 1 d -> b t d', b = b, t = t).mean(dim = 1, keepdim = True)

        return torch.cat((cls_tokens, out), dim = 1)

# contrastive learning functions

def model_forward_with_context(
    *,
    fn,
    args,
    freeze,
):
    encoding_context = nullcontext if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

# per-modality self-supervised loss wrapper

class ModalitySSL(nn.Module):
    def __init__(self, module, takes_mask = False):
        super().__init__()
        self.module = module
        self.takes_mask = takes_mask

    def forward(self, x, mask = None):
        if self.takes_mask:
            return self.module(x, mask = mask)

        return self.module(x)

# modality tower

class ModalityTower(nn.Module):
    def __init__(
        self,
        encoder,
        *,
        dim,
        dim_latent,
        pool = 'auto',
        has_cls_token = False,
        strip_cls_before_projection = False,
        mask_fn = None,
        postprocess_fn = None,
        pass_mask = False,
        to_latent = None,
        ssl_loss = None,
        ssl_loss_weight = 0.
    ):
        super().__init__()
        assert pool in {'auto', 'cls', 'mean', 'none'}, f'unknown pooling mode {pool}'

        self.encoder = encoder
        self.dim_latent = dim_latent
        self.pool = pool
        self.has_cls_token = has_cls_token
        self.strip_cls_before_projection = strip_cls_before_projection
        self.mask_fn = mask_fn
        self.postprocess_fn = postprocess_fn
        self.pass_mask = pass_mask

        self.to_latent = default(to_latent, nn.Linear(dim, dim_latent, bias = False))
        self.to_latent_extra = copy.deepcopy(self.to_latent)

        self.ssl_loss = ssl_loss
        self.ssl_loss_weight = ssl_loss_weight

    def encode(self, x, mask = None, freeze = False):
        args = (x, mask) if self.pass_mask and exists(mask) else (x,)
        enc = model_forward_with_context(fn = self.encoder, args = args, freeze = freeze)

        if exists(self.postprocess_fn):
            enc = self.postprocess_fn(enc, x)

        return enc

    def pool_latent(self, t):
        if t.ndim < 4 or self.pool == 'none':
            return t

        if self.pool == 'mean':
            return t.mean(dim = -2)

        return t[..., 0, :]

    def project(self, t, extra = False):
        proj = self.to_latent_extra if extra else self.to_latent
        return proj(t)

    def compute_ssl_loss(self, x, mask = None):
        return self.ssl_loss(x, mask)

# generic pairwise contrastive helpers

def pair_similarities(
    latents_a,
    latents_b,
    *,
    latents_a_extra = None,
    latents_b_extra = None,
    mask_a = None,
    mask_b = None,
    fine_grained = False
):
    if fine_grained:
        m, b, t = shape(latents_a, '[m] [b] [t] d')
        n, i = shape(latents_b, '[n] b [i] d')

        sim_a_to_b = einsum('m x t d, n y i d -> m n x y t i', latents_a, latents_b)

        mask_b_exp = maybe_rearrange(mask_b, '(n y) i -> 1 n 1 y 1 i', n = n)
        sim_a_to_b = mask_fill(sim_a_to_b, mask_b_exp)

        a_to_b = reduce(sim_a_to_b, '... t i -> ... t', 'max')

        mask_a_exp = maybe_rearrange(mask_a, '(m x) t -> m 1 x 1 t', m = m)
        a_to_b = masked_mean(a_to_b, mask_a_exp, dim = -1)
        a_to_b = rearrange(a_to_b, 'm n ... -> (m n) ...')

        latents_a_extra = default(latents_a_extra, latents_a)
        latents_b_extra = default(latents_b_extra, latents_b)

        sim_b_to_a = einsum('m x t d, n y i d -> m n x y i t', latents_a_extra, latents_b_extra)

        mask_a_exp = maybe_rearrange(mask_a, '(m x) t -> m 1 x 1 1 t', m = m)
        sim_b_to_a = mask_fill(sim_b_to_a, mask_a_exp)

        b_to_a = reduce(sim_b_to_a, '... i t -> ... i', 'max')

        mask_b_exp = maybe_rearrange(mask_b, '(n y) i -> 1 n 1 y i', n = n)
        b_to_a = masked_mean(b_to_a, mask_b_exp, dim = -1)
        b_to_a = rearrange(b_to_a, 'm n ... -> (m n) ...')

        return a_to_b, b_to_a

    sim_a_to_b = einsum('m i d, n j d -> m n i j', latents_a, latents_b)
    a_to_b = rearrange(sim_a_to_b, 'm n ... -> (m n) ...')

    latents_a_extra = default(latents_a_extra, latents_a)
    latents_b_extra = default(latents_b_extra, latents_b)

    sim_b_to_a = einsum('m i d, n j d -> m n j i', latents_a_extra, latents_b_extra)
    b_to_a = rearrange(sim_b_to_a, 'm n ... -> (m n) ...')

    return a_to_b, b_to_a

def intra_modal_off_diagonal(t):
    sim = einsum('m i d, m j d -> m i j', t, t)
    return off_diagonal(sim)

def mask_from_pad_id(x, pad_id):
    return x != pad_id

def reorder_eos(enc, text, eos_id):
    b, text_len = shape(text, '[b] [n]')

    eos_text_mask = (text == eos_id)
    assert torch.all(torch.any(eos_text_mask, dim = -1)), f'some of the text rows does not have the eos id {eos_id}'

    eos_indices = eos_text_mask.float().argmax(dim = -1, keepdim = True)
    eos_text_mask = torch.zeros_like(eos_text_mask).scatter(1, eos_indices, 1.).bool()
    eos_text_mask = rearrange(eos_text_mask, '... -> ... 1')

    eos_tokens = enc.masked_select(eos_text_mask)
    rest_tokens = enc.masked_select(~eos_text_mask)

    eos_tokens = rearrange(eos_tokens, '(b d) -> b 1 d', b = b)
    rest_tokens = rearrange(rest_tokens, '(b n d) -> b n d', b = b, n = text_len - 1)
    return torch.cat((eos_tokens, rest_tokens), dim = 1)

# generic multi-modal contrastive engine

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        image_encoder = None,
        text_encoder = None,
        modalities = None,
        extra_modalities = None,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        text_dim_head = 64,
        text_has_cls_token = True,
        text_pad_id = 0,
        text_rotary_pos_emb = False,
        text_causal_mask = False,
        text_eos_id = None,
        text_encode_without_mask = False,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_dim_head = 64,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_patch_dropout = 0.5,
        visual_has_cls_token = True,
        channels = 3,
        use_all_token_embeds = False,
        downsample_image_embeds = False,
        decoupled_contrastive_learning = False,
        extra_latent_projection = False,
        use_mlm = False,
        text_ssl_loss_weight = 0.05,
        use_visual_ssl = False,
        visual_ssl = None,
        visual_ssl_type = 'simsiam',
        visual_ssl_hidden_layer = -1,
        simclr_temperature = 0.1,
        image_ssl_loss_weight = 0.05,
        multiview_loss_weight = 0.1,
        checkpoint_during_training = False,
        sim_reg_loss_weight = 0.,
        spectral_loss_weight = 0.,
        asymmetric_infonce = False,
        logsumexp_reg_loss_weight = 0.,
        negative_mse_loss_weight = 0.,
        **kwargs
    ):
        super().__init__()

        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len
        self.text_encode_without_mask = text_encode_without_mask
        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        # engine settings

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        self.decoupled_contrastive_learning = decoupled_contrastive_learning
        self.extra_latent_projection = extra_latent_projection
        self.multiview_loss_weight = multiview_loss_weight

        # asymmetric infonce
        self.asymmetric_infonce = asymmetric_infonce

        # use the similarity regularization proposed in https://arxiv.org/abs/2309.08773
        self.sim_reg_loss_weight = sim_reg_loss_weight
        self.has_sim_reg_loss = sim_reg_loss_weight > 0.

        # use the spectral contrastive loss proposed in https://arxiv.org/abs/2106.04156
        self.spectral_loss_weight = spectral_loss_weight
        self.has_spectral_loss = spectral_loss_weight > 0.

        assert not (self.has_spectral_loss and use_all_token_embeds), (
            'spectral loss does not support fine-grained token embeddings'
        )

        # use the logsumexp regularizer proposed in https://arxiv.org/abs/2206.07568 (CRL)
        self.logsumexp_reg_loss_weight = logsumexp_reg_loss_weight
        self.has_logsumexp_reg_loss = logsumexp_reg_loss_weight > 0.

        assert not (self.has_logsumexp_reg_loss and not asymmetric_infonce), (
            'logsumexp regularizer requires asymmetric infonce'
        )

        # use an MSE penalty on the negative inner products
        self.negative_mse_loss_weight = negative_mse_loss_weight
        self.has_negative_mse_loss = negative_mse_loss_weight > 0.

        # is distributed or not
        self.requires_all_gather = distributed.is_initialized() and distributed.get_world_size() > 1

        # build modality towers

        configs = {}

        if exists(modalities):
            assert not extra_modalities, 'pass either modalities or extra_modalities, not both'
            configs = dict(modalities)
        else:
            extra_modalities = default(extra_modalities, {})

            assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), (
                'CLS token must be included on both vision and text transformers '
                'if not using fine-grained contrastive learning loss'
            )

            if 'text' not in extra_modalities:
                assert not (text_causal_mask and not exists(text_eos_id)), (
                    'text EOS token id must be given if using causal mask in text transformer'
                )

                text_transformer = text_encoder if exists(text_encoder) else TextTransformer(
                    dim = dim_text,
                    num_tokens = num_text_tokens + (1 if use_mlm else 0),
                    max_seq_len = text_seq_len,
                    depth = text_enc_depth,
                    heads = text_heads,
                    causal = text_causal_mask,
                    dim_head = text_dim_head,
                    rotary_pos_emb = text_rotary_pos_emb,
                    checkpoint_during_training = checkpoint_during_training
                )

                text_mask_fn = partial(mask_from_pad_id, pad_id = text_pad_id)
                text_postprocess_fn = partial(reorder_eos, eos_id = text_eos_id) if text_causal_mask else None

                text_ssl_loss = None
                text_ssl_weight = 0.

                if use_mlm:
                    mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)

                    mlm = MLM(
                        text_transformer,
                        dim = dim_text,
                        num_tokens = num_text_tokens,
                        **mlm_kwargs
                    )

                    text_ssl_loss = ModalitySSL(mlm, takes_mask = True)
                    text_ssl_weight = text_ssl_loss_weight

                configs['text'] = ModalityTower(
                    text_transformer,
                    dim = dim_text,
                    dim_latent = dim_latent,
                    pool = 'cls',
                    has_cls_token = text_has_cls_token,
                    mask_fn = text_mask_fn,
                    postprocess_fn = text_postprocess_fn,
                    pass_mask = not text_encode_without_mask,
                    ssl_loss = text_ssl_loss,
                    ssl_loss_weight = text_ssl_weight
                )

            if 'image' not in extra_modalities:
                visual_transformer = image_encoder if exists(image_encoder) else VisionTransformer(
                    dim = dim_image,
                    image_size = visual_image_size,
                    patch_size = visual_patch_size,
                    channels = channels,
                    depth = visual_enc_depth,
                    heads = visual_heads,
                    dim_head = visual_dim_head,
                    patch_dropout = visual_patch_dropout,
                    checkpoint_during_training = checkpoint_during_training
                )

                visual_ssl_loss = None
                visual_ssl_weight = 0.

                if use_visual_ssl or exists(visual_ssl):
                    if exists(visual_ssl):
                        visual_ssl_module = visual_ssl
                    else:
                        if visual_ssl_type == 'simsiam':
                            ssl_type = partial(SimSiam, channels = channels)
                        elif visual_ssl_type == 'simclr':
                            ssl_type = partial(SimCLR, temperature = simclr_temperature, channels = channels)
                        else:
                            raise ValueError(f'unknown visual_ssl_type')

                        visual_ssl_module = ssl_type(
                            visual_transformer,
                            image_size = visual_image_size,
                            hidden_layer = visual_ssl_hidden_layer
                        )

                    visual_ssl_loss = ModalitySSL(visual_ssl_module, takes_mask = False)
                    visual_ssl_weight = image_ssl_loss_weight

                to_visual_latent = None

                if downsample_image_embeds:
                    assert use_all_token_embeds, 'must use all token embeds for contrastive learning to downsample'

                    to_visual_latent = nn.Sequential(
                        RearrangeImage(),
                        nn.Conv2d(dim_image, dim_image, 4, stride = 2, padding = 1, bias = False, groups = dim_image),
                        nn.Conv2d(dim_image, dim_latent, 1),
                        Rearrange('b c h w -> b (h w) c')
                    )

                configs['image'] = ModalityTower(
                    visual_transformer,
                    dim = dim_image,
                    dim_latent = dim_latent,
                    pool = 'cls',
                    has_cls_token = visual_has_cls_token,
                    strip_cls_before_projection = downsample_image_embeds,
                    to_latent = to_visual_latent,
                    ssl_loss = visual_ssl_loss,
                    ssl_loss_weight = visual_ssl_weight
                )

            configs.update(extra_modalities)

        towers = {}

        for name, config in configs.items():
            assert isinstance(name, str) and name != '' and '.' not in name, f'invalid modality name {name}'

            if isinstance(config, ModalityTower):
                tower = config
            else:
                config = dict(config)
                config_dim_latent = config.pop('dim_latent', dim_latent)
                assert config_dim_latent == dim_latent, f'modality {name} latent dim must match {dim_latent}'

                tower = ModalityTower(dim_latent = config_dim_latent, **config)

            assert tower.dim_latent == dim_latent, f'modality {name} latent dim must match {dim_latent}'

            towers[name] = tower

        assert len(towers) > 0, 'at least one modality must be provided'

        self.modalities = nn.ModuleDict(towers)

        self.temperature = nn.Parameter(torch.tensor(1.))

    def forward(
        self,
        text = None,
        image = None,
        return_loss = False,
        return_encodings = False,
        return_latents = False,
        return_loss_breakdown = False,
        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
        freeze_text_encoder = False,    # text encoder is not trained if this is set to True
        text_to_image = True,           # direction for extra projection
        aug_text = None,                # augmented text (for multiview)
        aug_image = None,               # augmented image (for multiview)
        modalities = None,
        freeze_encoders = None
    ):
        assert not (return_loss_breakdown and not return_loss), 'return_loss must be True for breakdown'

        # gather inputs, keyed by modality name

        inputs = dict(modalities or {})

        if exists(text):
            inputs['text'] = text

        if exists(image):
            inputs['image'] = image

        if exists(aug_text):
            assert 'text' in inputs, 'augmented text passed in without text'
            inputs['text'] = (inputs['text'], *cast_tuple(aug_text))

        if exists(aug_image):
            assert 'image' in inputs, 'augmented image passed in without image'
            inputs['image'] = (inputs['image'], *cast_tuple(aug_image))

        modality_names = tuple(self.modalities.keys())

        assert len(inputs) > 0, 'at least one modality input must be provided'
        assert set(inputs).issubset(set(modality_names)), f'unknown modalities {set(inputs) - set(modality_names)}'

        active_names = tuple(name for name in modality_names if name in inputs)

        # normalize views per modality

        input_views = {name: cast_tuple(inputs[name]) for name in active_names}
        batch = size(input_views[active_names[0]][0], '[b] ...')
        num_views = {}

        for name, views in input_views.items():
            assert len(views) > 0, f'no views passed in for modality {name}'
            assert all(view.shape == views[0].shape for view in views), (
                f'all views for modality {name} must have the same shape'
            )
            assert_shape(views[0], 'b ...', b = batch)

            num_views[name] = len(views)

        batched_inputs = {
            name: torch.cat(views, dim = 0) if len(views) > 1 else views[0]
            for name, views in input_views.items()
        }

        is_multiview = any(views > 1 for views in num_views.values())

        assert not (return_loss and not self.training), 'loss cannot be used if not training'
        assert not (not return_loss and is_multiview), 'do not pass in augmented views if not training'
        assert not (self.multiview_loss_weight == 0 and is_multiview), (
            'multiview loss weight cannot be 0 if augmented views passed in'
        )

        if return_loss:
            assert len(active_names) >= 2, 'at least two modalities are required for contrastive loss'

        # freeze flags

        freeze_encoders = dict(default(freeze_encoders, {}))

        if freeze_text_encoder:
            freeze_encoders['text'] = True

        if freeze_image_encoder:
            freeze_encoders['image'] = True

        # encode

        encodings = {}
        masks = {}

        for name in active_names:
            tower = self.modalities[name]
            x = batched_inputs[name]
            mask = tower.mask_fn(x) if exists(tower.mask_fn) else None

            masks[name] = mask
            encodings[name] = tower.encode(x, mask = mask, freeze = freeze_encoders.get(name, False))

        # self supervised losses

        ssl_losses = {}

        if return_loss:
            for name in active_names:
                tower = self.modalities[name]

                if exists(tower.ssl_loss):
                    ssl_losses[name] = tower.compute_ssl_loss(batched_inputs[name], masks[name])

        # early return encodings

        if return_encodings:
            if len(active_names) == 1:
                return encodings[active_names[0]]

            if len(active_names) == 2:
                return tuple(encodings[name] for name in active_names)

            return encodings

        # project to latents and split views out

        latents = {}
        latents_extra = {}
        pooled_latents = {}
        pooled_latents_extra = {}

        for name in active_names:
            tower = self.modalities[name]
            enc = encodings[name]
            views = num_views[name]

            if tower.strip_cls_before_projection and tower.has_cls_token and enc.ndim == 3:
                enc = enc[:, 1:]

            latent = tower.project(enc)
            latent = rearrange(latent, '(v b) ... -> v b ...', v = views)

            pooled_latents[name] = tower.pool_latent(latent)

            strip_cls = self.use_all_token_embeds and tower.has_cls_token and not tower.strip_cls_before_projection

            if strip_cls and latent.ndim == 4:
                latent = latent[..., 1:, :]

            latents[name] = latent

            if self.extra_latent_projection:
                latent_extra = tower.project(enc, extra = True)
                latent_extra = rearrange(latent_extra, '(v b) ... -> v b ...', v = views)

                pooled_latents_extra[name] = tower.pool_latent(latent_extra)

                if strip_cls and latent_extra.ndim == 4:
                    latent_extra = latent_extra[..., 1:, :]

                latents_extra[name] = latent_extra
            else:
                latents_extra[name] = latent
                pooled_latents_extra[name] = pooled_latents[name]

        # early return latents

        if return_latents:
            def get_latent(name, extra = False):
                if self.use_all_token_embeds:
                    latent = latents_extra[name] if extra else latents[name]
                else:
                    latent = pooled_latents_extra[name] if extra else pooled_latents[name]

                return rearrange(latent, 'v b ... -> (v b) ...')

            if len(active_names) == 1:
                name = active_names[0]

                if self.extra_latent_projection:
                    return get_latent(name), get_latent(name, extra = True)

                return get_latent(name)

            if len(active_names) == 2:
                if self.extra_latent_projection:
                    base = tuple(get_latent(name) for name in active_names)
                    extra = tuple(get_latent(name, extra = True) for name in active_names)
                    return (*base, *extra)

                return tuple(get_latent(name) for name in active_names)

            if self.extra_latent_projection:
                return (
                    {name: get_latent(name) for name in active_names},
                    {name: get_latent(name, extra = True) for name in active_names}
                )

            return {name: get_latent(name) for name in active_names}

        # get temperature

        temp = self.temperature.exp()

        # early return similarities, if not returning loss

        if not return_loss:
            if len(active_names) == 2:
                a, b = active_names
                fine_grained = self.use_all_token_embeds and latents[a].ndim == 4 and latents[b].ndim == 4
                use_extra = self.extra_latent_projection and not text_to_image

                if fine_grained:
                    a_latent = latents_extra[a][0] if use_extra else latents[a][0]
                    b_latent = latents_extra[b][0] if use_extra else latents[b][0]

                    return einsum('b t d, b i d -> b t i', a_latent, b_latent) * temp

                a_latent = pooled_latents_extra[a][0] if use_extra else pooled_latents[a][0]
                b_latent = pooled_latents_extra[b][0] if use_extra else pooled_latents[b][0]

                return einsum('b d, b d -> b', a_latent, b_latent) * temp

            similarities = {}
            use_extra = self.extra_latent_projection and not text_to_image

            for a, b in combinations(active_names, 2):
                a_latent = pooled_latents_extra[a][0] if use_extra else pooled_latents[a][0]
                b_latent = pooled_latents_extra[b][0] if use_extra else pooled_latents[b][0]

                similarities[(a, b)] = einsum('b d, b d -> b', a_latent, b_latent) * temp

            return similarities

        # maybe distributed all gather

        if self.requires_all_gather:
            first_name = active_names[0]

            latents[first_name], sizes = all_gather(latents[first_name], 2, None)

            for name in active_names[1:]:
                latents[name], _ = all_gather(latents[name], 2, sizes)

            if self.extra_latent_projection:
                latents_extra[first_name], _ = all_gather(latents_extra[first_name], 2, sizes)

                for name in active_names[1:]:
                    latents_extra[name], _ = all_gather(latents_extra[name], 2, sizes)

            for name in active_names:
                mask = masks[name]

                if exists(mask):
                    views = num_views[name]
                    mask = rearrange(mask, '(v b) ... -> v b ...', v = views).to(torch.uint8)
                    mask, _ = all_gather(mask, 1, sizes)
                    masks[name] = rearrange(mask, 'v b ... -> (v b) ...').bool()

            for name in active_names:
                tower = self.modalities[name]
                pooled_latents[name] = tower.pool_latent(latents[name])

                if self.extra_latent_projection:
                    pooled_latents_extra[name] = tower.pool_latent(latents_extra[name])
                else:
                    pooled_latents_extra[name] = pooled_latents[name]

        # pairwise alignment over all modalities, as in AudioCLIP
        # https://arxiv.org/abs/2106.13043 -> generalized to n modalities, https://arxiv.org/abs/2403.12747

        pair_sims = {}
        pair_losses = []
        multiview_losses = []

        for a, b in combinations(active_names, 2):
            fine_grained = self.use_all_token_embeds and latents[a].ndim == 4 and latents[b].ndim == 4

            if fine_grained:
                a_latents, b_latents = latents[a], latents[b]
                a_latents_extra, b_latents_extra = latents_extra[a], latents_extra[b]
                a_mask, b_mask = masks[a], masks[b]
            else:
                a_latents, b_latents = pooled_latents[a], pooled_latents[b]
                a_latents_extra, b_latents_extra = pooled_latents_extra[a], pooled_latents_extra[b]
                a_mask, b_mask = None, None

            a_to_b, b_to_a = pair_similarities(
                a_latents,
                b_latents,
                latents_a_extra = a_latents_extra if self.extra_latent_projection else None,
                latents_b_extra = b_latents_extra if self.extra_latent_projection else None,
                mask_a = a_mask,
                mask_b = b_mask,
                fine_grained = fine_grained
            )

            pair_sims[(a, b)] = (a_to_b, b_to_a)

            a_to_b_loss = infonce_from_sim(a_to_b, temp, self.decoupled_contrastive_learning)
            b_to_a_loss = infonce_from_sim(b_to_a, temp, self.decoupled_contrastive_learning)
            pair_loss = average_directions(a_to_b_loss, b_to_a_loss, self.asymmetric_infonce)

            pair_losses.append(pair_loss[0])
            multiview_losses.extend(pair_loss[1:])

        cl_loss = torch.stack(pair_losses).mean()

        # maybe similarity regularize

        sim_reg_loss = 0.

        if self.has_sim_reg_loss:
            def averaged_intra_modal_sims(latent_dict):
                return {
                    name: intra_modal_off_diagonal(latent_dict[name]).mean(dim = 0)
                    for name in active_names
                }

            base_sims = averaged_intra_modal_sims(pooled_latents)
            base_losses = [F.mse_loss(base_sims[a], base_sims[b]) for a, b in combinations(active_names, 2)]

            if self.extra_latent_projection:
                extra_sims = averaged_intra_modal_sims(pooled_latents_extra)
                extra_losses = [F.mse_loss(extra_sims[a], extra_sims[b]) for a, b in combinations(active_names, 2)]
                sim_reg_loss = (torch.stack(base_losses).mean() + torch.stack(extra_losses).mean()) / 2
            else:
                sim_reg_loss = torch.stack(base_losses).mean()

        # maybe spectral contrastive loss

        spectral_loss = 0.

        if self.has_spectral_loss:
            spectral_loss = torch.stack([
                spectral_contrastive_loss(pooled_latents[a], pooled_latents[b])
                for a, b in combinations(active_names, 2)
            ]).mean()

        # maybe logsumexp regularize

        logsumexp_reg_loss = 0.

        if self.has_logsumexp_reg_loss:
            logsumexp_reg_loss = torch.stack([
                logsumexp_regularizer(a_to_b * temp)
                for a_to_b, _ in pair_sims.values()
            ]).mean()

        # maybe regularize negative inner products towards zero

        negative_mse_loss = 0.

        if self.has_negative_mse_loss:
            negative_mse_loss = torch.stack([
                average_directions(
                    off_diagonal_mse(a_to_b),
                    off_diagonal_mse(b_to_a),
                    self.asymmetric_infonce
                )
                for a_to_b, b_to_a in pair_sims.values()
            ]).mean()

        # multiview loss

        multiview_loss_weight = self.multiview_loss_weight if len(multiview_losses) > 0 else 0.
        multiview_cl_loss = torch.stack(multiview_losses).mean() if len(multiview_losses) > 0 else 0.

        # calculate weights

        ssl_loss_weight = sum(self.modalities[name].ssl_loss_weight for name in active_names)
        cl_loss_weight = 1 - (ssl_loss_weight + multiview_loss_weight)

        loss = cl_loss * cl_loss_weight

        for name, ssl_loss in ssl_losses.items():
            loss = loss + ssl_loss * self.modalities[name].ssl_loss_weight

        if len(multiview_losses) > 0:
            loss = loss + multiview_cl_loss * multiview_loss_weight

        # add similarity regularization loss with weight if needed

        if self.has_sim_reg_loss:
            loss = loss + sim_reg_loss * self.sim_reg_loss_weight

        # add spectral contrastive loss with weight if needed

        if self.has_spectral_loss:
            loss = loss + spectral_loss * self.spectral_loss_weight

        # add logsumexp regularizer with weight if needed

        if self.has_logsumexp_reg_loss:
            loss = loss + logsumexp_reg_loss * self.logsumexp_reg_loss_weight

        # add negative mse regularization with weight if needed

        if self.has_negative_mse_loss:
            loss = loss + negative_mse_loss * self.negative_mse_loss_weight

        if return_loss_breakdown:
            return Losses(
                loss,
                cl_loss,
                multiview_cl_loss,
                sim_reg_loss,
                spectral_loss,
                logsumexp_reg_loss,
                negative_mse_loss,
                ssl_losses
            )

        return loss
