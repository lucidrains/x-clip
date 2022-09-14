import math
import copy
from contextlib import contextmanager
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from x_clip.mlm import MLM
from x_clip.visual_ssl import SimSiam, SimCLR

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

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

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

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
    rot_dim = freqs.shape[-1]
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
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0., scale = 10.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
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

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

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
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

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
        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(self, x):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)

# contrastive learning functions

def model_forward_with_context(
    *,
    fn,
    args,
    freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

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
        **kwargs
    ):
        super().__init__()
        assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'

        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_encode_without_mask = text_encode_without_mask # whether to pass in text mask to text encoder

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
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
                heads = visual_heads,
                dim_head = visual_dim_head,
                checkpoint_during_training = checkpoint_during_training
            )

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        if use_mlm:
            mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)

            self.mlm = MLM(
                self.text_transformer,
                dim = dim_text,
                num_tokens = num_text_tokens,
                **mlm_kwargs
            )

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        if self.use_visual_ssl:
            if exists(visual_ssl):
                self.visual_ssl = visual_ssl

            elif use_visual_ssl:
                if visual_ssl_type == 'simsiam':
                    ssl_type = partial(SimSiam, channels = channels)
                elif visual_ssl_type == 'simclr':
                    ssl_type = partial(SimCLR, temperature = simclr_temperature, channels = channels)
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

        self.multiview_loss_weight = multiview_loss_weight

    def forward(
        self,
        text,
        image,
        return_loss = False,
        return_encodings = False,
        return_latents = False,
        freeze_image_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
        freeze_text_encoder = False,    # text encoder is not trained if this is set to True
        text_to_image = True,           # in the case the extra projection is turned on, would return different similarity values depending on modality directionality
        aug_text = None,                # augmented text (for multiview)
        aug_image = None                # augmented image (for multiview)
    ):
        b, device = text.shape[0], text.device

        # derive text mask

        text_mask = text != self.text_pad_id

        # ssl

        text_ssl_loss = 0
        image_ssl_loss = 0

        if return_loss:
            text_ssl_loss = self.mlm(text, mask = text_mask) if self.use_mlm else 0
            image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0

        # concat augmented texts and images and do some asserts

        num_batch_texts = num_batch_images = 1

        if exists(aug_text):
            aug_text = cast_tuple(aug_text)
            assert all(map(lambda t: t.shape == text.shape, aug_text))
            num_batch_texts = len(aug_text) + 1

            aug_text = torch.cat(aug_text, dim = 0)

            aug_text_mask = aug_text != self.text_pad_id

            text_mask = torch.cat((text_mask, aug_text_mask), dim = 0)
            text = torch.cat((text, aug_text), dim = 0)

        if exists(aug_image):
            aug_image = cast_tuple(aug_image)
            assert all(map(lambda i: i.shape == image.shape, aug_image))
            num_batch_images = len(aug_image) + 1

            aug_image = torch.cat(aug_image, dim = 0)

            image = torch.cat((image, aug_image), dim = 0)

        is_multiview = (num_batch_texts > 1 or num_batch_images > 1)
        assert not (return_loss and not self.training), 'loss cannot be used if not training'
        assert not (not return_loss and is_multiview), 'do not pass in augmented texts or images if not training'
        assert not (self.multiview_loss_weight == 0 and is_multiview), 'multiview loss weight cannot be 0 if augmented text or images passed in'

        # get encoded text

        text_args = (text,)

        if not self.text_encode_without_mask:
            text_args = (*text_args, text_mask)

        enc_text = model_forward_with_context(
            fn = self.text_transformer,
            args = text_args,
            freeze = freeze_text_encoder
        )

        # depending on whether text is using causal mask, post process, moving eos token to the first position

        if self.text_causal_mask:
            eos_text_mask = (text == self.text_eos_id)
            assert torch.all(torch.any(eos_text_mask, dim = -1)), f'some of the text rows does not have the eos id {self.text_eos_id}'

            text_len = text.shape[-1]
            eos_indices = eos_text_mask.float().argmax(dim = -1, keepdim = True)

            eos_text_mask = torch.zeros_like(eos_text_mask).scatter(1, eos_indices, 1.).bool()
            eos_text_mask = rearrange(eos_text_mask, '... -> ... 1')

            eos_tokens = enc_text.masked_select(eos_text_mask)
            rest_tokens = enc_text.masked_select(~eos_text_mask)

            eos_tokens = rearrange(eos_tokens, '(b d) -> b 1 d', b = b)
            rest_tokens = rearrange(rest_tokens, '(b n d) -> b n d', b = b, n = text_len - 1)
            enc_text = torch.cat((eos_tokens, rest_tokens), dim = 1)

        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT

        enc_image = model_forward_with_context(
            fn = self.visual_transformer,
            args = (image,),
            freeze = freeze_image_encoder
        )

        # early return of encodings, if needed (for DALL-E2)

        if return_encodings:
            return enc_text, enc_image

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        if self.use_all_token_embeds:
            assert enc_text.ndim == 3, 'encoded text must have 3 dimensions (batch, seq, features)'
            assert enc_image.ndim == 3, 'encoded image must have 3 dimensions (batch, seq [height x width], features)'
            text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
            image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
        else:
            text_embeds = enc_text[:, 0] if enc_text.ndim == 3 else enc_text
            image_embeds = enc_image[:, 0] if enc_image.ndim == 3 else enc_image

        # project to latents

        text_latents = self.to_text_latent(text_embeds)
        image_latents = self.to_visual_latent(image_embeds)
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # calculate another set of latents for image to text (vs text to image)
        # proposed by CLOOB

        text_latents_extra, image_latents_extra = text_latents, image_latents
        if self.extra_latent_projection:
            text_latents_extra = self.to_text_latent_extra(text_embeds)
            image_latents_extra = self.to_visual_latent_extra(image_embeds)
            text_latents_extra, image_latents_extra = map(l2norm, (text_latents_extra, image_latents_extra))

        # whether to early return latents

        if return_latents:
            if self.extra_latent_projection:
                return text_latents, image_latents, text_latents_extra, image_latents_extra

            return text_latents, image_latents

        # get temperature

        temp = self.temperature.exp()

        # early return, if needed

        if not return_loss and self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b t d, b i d -> b t i', *einsum_args) * temp

        if not return_loss and not self.use_all_token_embeds:
            einsum_args = (text_latents_extra, image_latents_extra) if self.extra_latent_projection and not text_to_image else (text_latents, image_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp

        # split out multiview dimension for text and images

        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts)
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images)

        if self.extra_latent_projection:
            text_latents_extra = rearrange(text_latents_extra, '(m b) ... -> m b ...', m = num_batch_texts)
            image_latents_extra = rearrange(image_latents_extra, '(m b) ... -> m b ...', m = num_batch_images)

        # contrastive loss

        """
        m - num batches of text (for multiview)
        n - num batches of images (for multiview)
        x - batches of text
        y - batches of images
        t - sequence dimension along text tokens
        i - sequence dimension along image tokens
        """

        if self.use_all_token_embeds:
            # fine-grained CLIP logic
            sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_latents, image_latents) * temp

            sim_image_to_text = sim_text_to_image
            if self.extra_latent_projection:
                sim_image_to_text = einsum('m x t d, n y i d -> m n x y t i', text_latents_extra, image_latents_extra) * temp

            text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
            text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m = num_batch_texts)
            text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

            image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
            masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
            image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')
        else:
            text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
            image_to_text = rearrange(text_to_image, '... t i -> ... i t')

            if self.extra_latent_projection:
                image_to_text = einsum('m t d, n i d -> m n i t', text_latents_extra, image_latents_extra) * temp

        # calculate loss

        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # exponentiate

        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # numerators

        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        # denominator

        if self.decoupled_contrastive_learning:
            pos_mask = torch.eye(b, device = device, dtype = torch.bool)
            text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

        # loss

        text_to_image_loss = -log(text_to_image_pos / text_to_image_denom).mean(dim = -1)
        image_to_text_loss = -log(image_to_text_pos / image_to_text_denom).mean(dim = -1)

        # calculate CL loss

        cl_losses = (text_to_image_loss + image_to_text_loss) / 2

        # get main CL loss vs multiview CL losses

        cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

        # if no augmented text or images passed in, multiview loss weight is 0

        multiview_loss_weight = self.multiview_loss_weight if is_multiview else 0

        # calculate weights

        cl_loss_weight = 1 - (self.text_ssl_loss_weight + self.image_ssl_loss_weight + multiview_loss_weight)

        loss = (cl_loss * cl_loss_weight) \
            + (text_ssl_loss * self.text_ssl_loss_weight) \
            + (image_ssl_loss * self.image_ssl_loss_weight)

        # add multiview CL loss with weight

        if is_multiview:
            loss = loss + multiview_cl_loss.mean() * multiview_loss_weight

        return loss
