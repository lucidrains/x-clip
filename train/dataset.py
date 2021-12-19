 # Based on:
 # https://github.com/Zasder3/open_clip_juwels/blob/d36754b624a3eb5f0513ae3d0ee4030a420409e5/src/training/data.py
 # https://github.com/Zasder3/open_clip_juwels/blob/50308cffdb4cf1b41c1fe95d8e8f8665c6a5c5d6/src/clip/clip.py

from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from x_clip.tokenizer import tokenizer
import os
import braceexpand
import math
import webdataset as wds
from datetime import datetime


def preprocess_txt(text):
    return tokenizer.tokenize(text)


def _convert_to_rgb(image):
    return image.convert('RGB')


def transform_img(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    if 'sizes.json' in os.listdir(dir_path):
        sizes_filename = os.path.join(dir_path, 'sizes.json')
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum(
            [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif '__len__' in os.listdir(dir_path):
        total_size = eval(open(os.path.join(dir_path, '__len__'), 'r').read())
    else:
        raise ValueError(f'Could not find dataset size in {dir_path}')
    num_shards = len(shards_list)
    return total_size, num_shards


def get_wds_dataset(args, is_train=True, logger=None):
    input_shards = args.path_data_train
    assert input_shards is not None

    # The following code is adapted from https://github.com/tmbdev/webdataset-examples/blob/master/main-wds.py
    num_samples, num_shards = get_dataset_size(input_shards)

    max_shards_per_node = math.ceil(num_shards / args.world_size)
    num_samples = args.world_size * (num_samples * max_shards_per_node // num_shards)
    num_batches = num_samples // (args.bs * args.world_size)
    num_samples = num_batches * args.bs * args.world_size

    logger.info(f"{datetime.now()} rank: {args.rank} max_shards_per_node: {max_shards_per_node}")
    logger.info(f"{datetime.now()} rank: {args.rank} num_batches: {num_batches}")
    logger.info(f"{datetime.now()} rank: {args.rank} num_samples: {num_samples}")

    shardlist = wds.PytorchShardList(
        input_shards,
        epoch_shuffle=is_train,
        split_by_node=is_train  # NOTE: we do eval on a single gpu.
    )

    preprocess_img = transform_img(args.visual_image_size, True)

    dataset = (
        wds.WebDataset(shardlist)
        .decode("pil")
        .rename(image="jpg;png", text="txt")
        .map_dict(image=preprocess_img, text=preprocess_txt)
        .to_tuple("image", "text")
        #.batched(args.bs, partial=not is_train or not args.distributed)
        .batched(args.bs, partial=not is_train)
    )

    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.numw,
    )

    # With DDP, we need to make sure that all nodes get the same number of batches;
    # we do that by reusing a little bit of data.
    dataloader = dataloader.repeat(2).slice(num_batches)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader
