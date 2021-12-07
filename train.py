import argparse
import os

import torch
import torchvision.transforms.functional as tvt_fn
import tqdm
from torch.cuda.amp import autocast

import wandb
import x_clip


def debug(message: str, quiet: bool = False, *args, **kwargs):
    if quiet:
        print(message, *args, **kwargs)


def get_argparser():
    p = argparse.ArgumentParser(description='Train a CLIP model')
    p.add_argument('--datadir', type=str, default='data/', help='Path to the dataset')
    p.add_argument('--bpe_path', type=str)
    p.add_argument('--batch-size', type=int, default=64, help='Batch size')
    p.add_argument('--num-epochs', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--clip_grad_norm_factor', type=float, default=1.0)
    p.add_argument('--log_frequency', type=int, default=10)
    p.add_argument('--ckpt_save_path', type=str, default='./checkpoints/x_clip.pt')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing checkpoints')
    p.add_argument('--amp', action='store_true', help='Enable AMP')
    p.add_argument('--dim_text', type=int, default=512)
    p.add_argument('--dim_image', type=int, default=512)
    p.add_argument('--dim_latent', type=int, default=512)
    p.add_argument('--text_enc_depth', type=int, default=6)
    p.add_argument('--text_seq_len', type=int, default=256)
    p.add_argument('--text_heads', type=int, default=8)
    p.add_argument('--num_visual_tokens', type=int, default=512)
    p.add_argument('--visual_enc_depth', type=int, default=6)
    p.add_argument('--visual_heads', type=int, default=8)
    p.add_argument('--visual_image_size', type=int, default=256)
    p.add_argument('--visual_patch_size', type=int, default=32)
    p.add_argument('--channels', type=int, default=3)
    p.add_argument('--use_all_token_embeds', action='store_true')
    return p


def clip_model_from_args(args, num_text_tokens):
    return x_clip.CLIP(
        num_text_tokens=num_text_tokens,
        dim_text=args.dim_text,
        dim_image=args.dim_image,
        dim_latent=args.dim_latent,
        text_enc_depth=args.text_enc_depth,
        text_heads=args.text_heads,
        num_visual_tokens=args.num_visual_tokens,
        visual_enc_depth=args.visual_enc_depth,
        visual_heads=args.visual_heads,
        visual_image_size=args.visual_image_size,
        visual_patch_size=args.visual_patch_size,
        channels=args.channels,
        use_all_token_embeds=args.use_all_token_embeds,
        text_seq_len=args.text_seq_len,
    )


def check_args(args):
    if not os.path.exists(args.bpe_path):
        raise ValueError('BPE path does not exist')
    if not os.path.exists(args.datadir):
        raise ValueError('Dataset path does not exist')
    if os.path.exists(args.ckpt_save_path) and not args.overwrite:
        raise ValueError(
            'Checkpoint path already exists, use --overwrite to overwrite')
    else:
        os.makedirs(os.path.dirname(args.ckpt_save_path), exist_ok=True)


def main():
    parser = get_argparser()
    args = parser.parse_args()
    check_args(args)

    wandb_run = wandb.init(project='x-clip', config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yttm_tokenizer = x_clip.YttmTokenizer(args.bpe_path)

    clip_model = clip_model_from_args(args, yttm_tokenizer.vocab_size)
    clip_model.to(device)
    clip_model.train()

    # clamp to ln(100)
    torch.clamp(clip_model.temperature, min=0.0, max=4.6052)

    def preprocess_fn(x):
        return tvt_fn.resize(tvt_fn.to_tensor(x),
                             (args.visual_image_size, args.visual_image_size))

    image_text_dataset = x_clip.ImageTextDataset(preprocess=preprocess_fn,
                                          folder=args.datadir,
                                          bpe_tokenizer=yttm_tokenizer)
    image_text_dataloader = torch.utils.data.DataLoader(
        image_text_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    # Create optimizer
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=args.learning_rate)

    # Train
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        current_epoch_pbar = tqdm.tqdm(enumerate(image_text_dataloader), total=len(image_text_dataloader), unit='batch', unit_scale=args.batch_size)
        for batch_idx, (text, images) in current_epoch_pbar:
            with autocast(enabled=args.amp):
                text, images = map(lambda t: t.cuda(), (text, images))
                mask = torch.ones_like(text).bool()
                loss = clip_model(text, images, text_mask=mask, return_loss=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), args.clip_grad_norm_factor)
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % args.log_frequency == 0:
                current_epoch_pbar.set_description(f'batch {batch_idx} loss {loss:.4f}')
                wandb_run.log({'loss': loss, 'epoch': epoch}, step=batch_idx)

        torch.save(clip_model.state_dict(), args.ckpt_save_path)


if __name__ == "__main__":
    main()
