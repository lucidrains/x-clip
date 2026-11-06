import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from x_clip import CLIP, TextTransformer, VisionTransformer, ModalityTower, Losses
from x_clip.x_clip import spectral_contrastive_loss

# fixtures

@pytest.fixture(autouse = True)
def set_seed():
    torch.manual_seed(0)

def make_clip(**kwargs):
    return CLIP(
        dim_text = 32,
        dim_image = 32,
        dim_latent = 16,
        num_text_tokens = 100,
        text_enc_depth = 1,
        text_seq_len = 8,
        text_heads = 2,
        visual_enc_depth = 1,
        visual_image_size = 8,
        visual_patch_size = 4,
        visual_heads = 2,
        visual_patch_dropout = 0.1,
        **kwargs
    )

def make_data(batch = 4):
    text = torch.randint(0, 100, (batch, 8))
    images = torch.randn(batch, 3, 8, 8)
    return text, images

class AudioEncoder(nn.Module):
    def __init__(self, dim = 32, input_dim = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, audio):
        return self.net(audio)

def make_trilinear_data(batch = 4):
    text = torch.randint(0, 100, (batch, 8))
    video = torch.randn(batch, 2, 3, 8, 8)
    audio = torch.randn(batch, 16)
    return text, video, audio

def make_extra_modality(dim = 32, input_dim = 16):
    return {
        'encoder': AudioEncoder(dim = dim, input_dim = input_dim),
        'dim': dim,
        'pool': 'none'
    }

def make_trilinear_clip(**kwargs):
    return make_clip(
        extra_modalities = {'audio': make_extra_modality()},
        **kwargs
    )

def make_quadmodal_clip(**kwargs):
    return make_clip(
        extra_modalities = {
            'audio': make_extra_modality(),
            'depth': make_extra_modality()
        },
        **kwargs
    )

# e2e tests

@pytest.mark.parametrize('spectral_loss_weight', (0., 0.1))
@pytest.mark.parametrize('asymmetric_infonce', (False, True))
@pytest.mark.parametrize('logsumexp_reg_loss_weight', (0., 0.01))
@pytest.mark.parametrize('negative_mse_loss_weight', (0., 0.1))
def test_x_clip_e2e(
    spectral_loss_weight,
    asymmetric_infonce,
    logsumexp_reg_loss_weight,
    negative_mse_loss_weight
):
    if logsumexp_reg_loss_weight > 0. and not asymmetric_infonce:
        pytest.skip('logsumexp regularizer requires asymmetric infonce')

    clip = make_clip(
        spectral_loss_weight = spectral_loss_weight,
        asymmetric_infonce = asymmetric_infonce,
        logsumexp_reg_loss_weight = logsumexp_reg_loss_weight,
        negative_mse_loss_weight = negative_mse_loss_weight
    )

    text, images = make_data()
    loss = clip(text, images, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in clip.parameters())

@pytest.mark.parametrize('extra_latent_projection', (False, True))
@pytest.mark.parametrize('decoupled_contrastive_learning', (False, True))
def test_x_clip_e2e_projection_and_dcl(extra_latent_projection, decoupled_contrastive_learning):
    clip = make_clip(
        extra_latent_projection = extra_latent_projection,
        decoupled_contrastive_learning = decoupled_contrastive_learning,
        spectral_loss_weight = 0.1,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, images = make_data()
    loss = clip(text, images, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()

@pytest.mark.parametrize('aug_text', (False, True))
@pytest.mark.parametrize('aug_image', (False, True))
def test_x_clip_multiview_e2e(aug_text, aug_image):
    clip = make_clip(
        extra_latent_projection = True,
        multiview_loss_weight = 0.1,
        spectral_loss_weight = 0.1,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, images = make_data()
    aug_text = torch.randint(0, 100, (4, 8)) if aug_text else None
    aug_image = torch.randn(4, 3, 8, 8) if aug_image else None

    loss = clip(text, images, aug_text = aug_text, aug_image = aug_image, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()

def test_x_clip_trilinear_text_video_audio():
    clip = make_trilinear_clip(
        extra_latent_projection = True,
        spectral_loss_weight = 0.1,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, video, audio = make_trilinear_data()
    loss = clip(text, video, modalities = {'audio': audio}, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in clip.parameters())
    assert clip.modalities['audio'].to_latent.weight.grad is not None

def test_x_clip_trilinear_inference():
    clip = make_trilinear_clip(extra_latent_projection = True)
    clip.eval()

    text, video, audio = make_trilinear_data()

    with torch.no_grad():
        sims = clip(text, video, modalities = {'audio': audio})

    assert set(sims.keys()) == {('text', 'image'), ('text', 'audio'), ('image', 'audio')}
    assert all(sim.shape == (4,) for sim in sims.values())

def test_x_clip_trilinear_return_latents():
    clip = make_trilinear_clip(extra_latent_projection = True)
    clip.eval()

    text, video, audio = make_trilinear_data()

    with torch.no_grad():
        base_latents, extra_latents = clip(text, video, modalities = {'audio': audio}, return_latents = True)

    expected = {'text', 'image', 'audio'}
    assert set(base_latents.keys()) == expected
    assert set(extra_latents.keys()) == expected
    assert all(latent.shape == (4, 16) for latent in base_latents.values())
    assert all(latent.shape == (4, 16) for latent in extra_latents.values())

def test_x_clip_trilinear_fine_grained():
    clip = make_trilinear_clip(
        use_all_token_embeds = True,
        extra_latent_projection = True,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, video, audio = make_trilinear_data()
    loss = clip(text, video, modalities = {'audio': audio}, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()

def test_x_clip_quadmodal_text_video_audio_depth():
    clip = make_quadmodal_clip(
        extra_latent_projection = True,
        spectral_loss_weight = 0.1,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, video, audio = make_trilinear_data()
    depth = torch.randn(4, 16)

    loss = clip(
        text,
        video,
        modalities = {'audio': audio, 'depth': depth},
        return_loss = True
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert clip.modalities['audio'].to_latent.weight.grad is not None
    assert clip.modalities['depth'].to_latent.weight.grad is not None

def test_x_clip_fully_custom_modalities():
    caption_encoder = TextTransformer(
        dim = 32,
        num_tokens = 100,
        max_seq_len = 8,
        depth = 1,
        heads = 2,
        dim_head = 64
    )

    video_encoder = VisionTransformer(
        dim = 32,
        image_size = 8,
        patch_size = 4,
        channels = 3,
        depth = 1,
        heads = 2,
        dim_head = 64,
        patch_dropout = 0.1
    )

    clip = CLIP(
        modalities = {
            'caption': ModalityTower(
                caption_encoder,
                dim = 32,
                dim_latent = 16,
                pool = 'cls',
                has_cls_token = True,
                mask_fn = lambda x: x != 0,
                pass_mask = True
            ),
            'video': ModalityTower(
                video_encoder,
                dim = 32,
                dim_latent = 16,
                pool = 'cls',
                has_cls_token = True
            ),
            'audio': ModalityTower(
                AudioEncoder(dim = 32),
                dim = 32,
                dim_latent = 16,
                pool = 'none'
            ),
            'depth': ModalityTower(
                AudioEncoder(dim = 32),
                dim = 32,
                dim_latent = 16,
                pool = 'none'
            )
        },
        dim_latent = 16,
        extra_latent_projection = True,
        spectral_loss_weight = 0.1,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, video, audio = make_trilinear_data()
    depth = torch.randn(4, 16)

    loss = clip(
        modalities = {
            'caption': text,
            'video': video,
            'audio': audio,
            'depth': depth
        },
        return_loss = True
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in clip.parameters())

def test_x_clip_fine_grained_e2e():
    clip = make_clip(
        use_all_token_embeds = True,
        extra_latent_projection = True,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, images = make_data()
    loss = clip(text, images, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()

def test_spectral_contrastive_learning_rejects_fine_grained():
    with pytest.raises(AssertionError):
        make_clip(use_all_token_embeds = True, spectral_loss_weight = 0.1)

def test_logsumexp_regularizer_requires_asymmetric_infonce():
    with pytest.raises(AssertionError):
        make_clip(logsumexp_reg_loss_weight = 0.01)

# loss breakdown

def test_x_clip_loss_breakdown():
    clip = make_clip(
        spectral_loss_weight = 0.1,
        asymmetric_infonce = True,
        logsumexp_reg_loss_weight = 0.01,
        negative_mse_loss_weight = 0.1
    )

    text, images = make_data()
    losses = clip(text, images, return_loss = True, return_loss_breakdown = True)

    assert isinstance(losses, Losses)
    assert losses._fields == Losses._fields

    expected = losses.contrastive \
        + losses.spectral * 0.1 \
        + losses.logsumexp * 0.01 \
        + losses.negative_mse * 0.1

    assert torch.allclose(losses.loss, expected)

def test_x_clip_loss_breakdown_requires_loss():
    clip = make_clip()
    text, images = make_data()

    with pytest.raises(AssertionError):
        clip(text, images, return_loss = False, return_loss_breakdown = True)

# loss formula

def test_spectral_contrastive_loss_formula():
    text_latents = F.normalize(torch.randn(2, 4, 8), dim = -1)
    image_latents = F.normalize(torch.randn(2, 4, 8), dim = -1)

    loss = spectral_contrastive_loss(text_latents, image_latents)

    sim = torch.einsum('m i d, n j d -> m n i j', text_latents, image_latents)
    pos = torch.diagonal(sim, dim1 = -2, dim2 = -1).mean()

    neg_mask = ~torch.eye(4, dtype = torch.bool)
    neg = sim[..., neg_mask]

    expected = -2 * pos + (neg ** 2).mean()
    assert torch.allclose(loss, expected)

# rotary positional embeddings on causal (cls-token-free) text encoders

def test_text_transformer_causal_rotary_pos_emb():
    transformer = TextTransformer(
        dim = 32,
        num_tokens = 100,
        max_seq_len = 8,
        depth = 1,
        heads = 2,
        dim_head = 64,
        causal = True,
        rotary_pos_emb = True
    )

    text = torch.randint(0, 100, (2, 8))
    out = transformer(text)

    assert out.shape == (2, 8, 32)

def test_x_clip_causal_rotary_text():
    clip = make_clip(
        text_rotary_pos_emb = True,
        text_causal_mask = True,
        text_eos_id = 1
    )

    text, images = make_data()
    text[:, -1] = 1

    loss = clip(text, images, return_loss = True)

    assert torch.isfinite(loss)
    loss.backward()
