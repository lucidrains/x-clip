import pytest
import torch
import torch.nn.functional as F

from x_clip import CLIP, Losses
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
