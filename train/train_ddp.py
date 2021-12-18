import argparse
import os
import sys
import json
import time
from datetime import datetime
import logging
from functools import partial
import torch
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import AdamW
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.profiler

from x_clip import CLIP
from x_clip import tokenizer as simple_tokenizer


# multi-GPU training script based on https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

# Example command to start a training run:
# TO DO: Add example here!


def get_args():
    """Get all parsed arguments."""
    parser = argparse.ArgumentParser(description="X-CLIP ddp training")

    # data
    parser.add_argument("--id", type=str,
                        help="run id")

    # TO DO: Add input data options here!

    parser.add_argument("--path-results", type=str, default="results",
                        help="path to the results data, i.e., logs, model weights, etc. (default: results)")
    parser.add_argument("--path-weights", type=str, default=None,
                        help="path to weights for reloading (default: None)")
    parser.add_argument("--numw", type=int, default=0,
                        help="number of workers for pytorch dataloader (default: 0)")

    # training
    parser.add_argument("--lr", type=int, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--bs", type=int, default=8,
                        help="batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="epochs (default: 2)")
    parser.add_argument("--dryrun", type=int, default=None,
                        help="Run dryrun steps per epoch to test the setup (default: None)")

    # X-CLIP model
    parser.add_argument("--dim-text", type=int, default=512,
                        help="text encoder dim_text (default: 512)")
    parser.add_argument("--dim-image", type=int, default=512,
                        help="image encoder dim_image (default: 512)")
    parser.add_argument("--dim-latent", type=int, default=512,
                        help="dim_latent (default: 512)")
    parser.add_argument("--num-text-tokens", type=int, default=512,
                        help="num_text_tokens (default: 512)")
    parser.add_argument("--num-visual-tokens", type=int, default=512,
                        help="num_visual_tokens (default: 512)")
    parser.add_argument("--text-enc-depth", type=int, default=6,
                        help="text_enc_depth (default: 6)")
    parser.add_argument("--text-seq-len", type=int, default=256,
                        help="text_seq_len (default: 256)")
    parser.add_argument("--text-heads-len", type=int, default=8,
                        help="text_heads (default: 8)")
    parser.add_argument("--num-visual-tokens", type=int, default=512,
                        help="num_visual_tokens (default: 512)")
    parser.add_argument("--visual-enc-depth", type=int, default=6,
                        help="visual_enc_depth (default: 6)")
    parser.add_argument("--visual-enc-depth", type=int, default=6,
                        help="visual_enc_depth (default: 6)")
    parser.add_argument("--visual-heads", type=int, default=8,
                        help="visual_heads (default: 8)")
    parser.add_argument("--visual-image-size", type=int, default=256,
                        help="visual_image_size (default: 256)")
    parser.add_argument("--visual-patch-size", type=int, default=32,
                        help="visual_patch_size (default: 32)")
    parser.add_argument("--channels", type=int, default=3,
                        help="channels (default: 3)")
    parser.add_argument("--use-all-token-embeds", action="store_true", default=False,
                        help="use_all_token_embeds (default: False)")
    parser.add_argument("--downsample-image-embeds", action="store_true", default=False,
                        help="downsample_image_embeds (default: False)")
    parser.add_argument("--decoupled-contrastive-learning", action="store_true", default=False,
                        help="decoupled_contrastive_learning (default: False)")
    parser.add_argument("--extra-latent-projection", action="store_true", default=False,
                        help="extra_latent_projection (default: False)")
    parser.add_argument("--return-loss", action="store_true", default=True,
                        help="return_loss (default: True)")
    parser.add_argument("--freeze-image-encoder", action="store_true", default=False,
                        help="freeze_image_encoder: False)")
    parser.add_argument("--text-to-image", action="store_true", default=True,
                        help="text_to_image default: True)")
    parser.add_argument("--clip-grad-norm", type=float, default=None,
                        help="clip_grad_norm (default: None)")

    # logging and saving
    parser.add_argument("--save-interval-epoch", type=int, default=1,
                        help="save interval epoch (default: 1")
    parser.add_argument("--save-interval-step", type=int, default=4_000,
                        help="save interval step (default: 4_000")
    parser.add_argument("--tb-profiler", action="store_true", default=False,
                        help="run with tb profiler (default: False)")

    # TO DO: Add option to resume from weight path!

    args = parser.parse_args()
    args.cmd = " ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv) # log the exact python command for the run
    return args


def create_logger(path_log, file_name):
    file_path = os.path.join(path_log, file_name)

    # file handler for logging to file
    file_handler = logging.filehandler(file_path)
    file_handler.setlevel(logging.debug)

    # console handler for logging to the console
    console_handler = logging.streamhandler()
    console_handler.setlevel(logging.info)

    logger = logging.getlogger("logger")
    logger.setlevel(logging.debug)
    logger.propagate = False
    if file_path is not none:
        logger.addhandler(file_handler)
    logger.addhandler(console_handler)

    return logger


def setup(rank, world_size):
    os.environ['master_addr'] = "localhost"
    os.environ['master_port'] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # TO DO: Check nccl backend with newest PyTorch version (because of the all gather bug in a older version).


def cleanup():
    dist.destroy_process_group()


class AverageMeter(object):
    """computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# from https://github.com/rwightman/pytorch-image-models/blob/779107b693010934ac87c8cecbeb65796e218488/timm/utils/distributed.py#L11
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def train_ddp(args, model, optimizer, dl_train, dl_valid_id, dl_valid_ood, epochs, logger=None, writer=None):

    # Based on: https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    # TO DO: Check if still needed with latest PyTorch version.
    torch.cuda.set_device(args.rank)
    torch.cuda.empty_cache()

    step = 0

    logger.info(f"{datetime.now()} rank: {args.rank} world_size: {args.world_size}")
    setup(args.rank, args.world_size)
    logger.info(f"{datetime.now()} rank: {args.rank} ddp setup")
    model.to(args.rank)
    logger.info(f"{datetime.now()} rank: {args.rank} model moved to rank {args.rank}")
    ddp_model = DDP(model, device_ids=[args.rank])
    logger.info(f"{datetime.now()} rank: {args.rank} created ddp model")

    def one_epoch(args, model, optimizer, dl_train, dl_valid, epoch, step):
        time_epoch_start = time.time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        model.train()

        if args.tb_profile:
            prof = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_path),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                    )
            prof.start()

        tp = time.time()
        for i, b in enumerate(dl_train):

            if args.dryrun:
                if i == args.dryrun:
                    break

            #optimizer.zero_grad()
            # Faster option:
            for param in model.parameters():
                param.grad = None

            dt = time.time() - tp
            dt = torch.tensor(dt).to(args.rank)
            dt = reduce_tensor(dt, args.world_size)
            data_time.update(dt)

            text, text_mask, image = b

            text      = text.to(args.rank)
            text_mask = text_mask.to(args.rank)
            image     = image.to(args.rank)

            loss = model(
                    text,
                    image,
                    text_mask = text_mask,
                    return_loss = args.return_loss,
                    freeze_image_encoder = args.freeze_image_encoder,
                    text_to_image = args.text_to_image,
                    )

            # TO DO: Check if barrier is needed.
            dist.barrier()
            loss.backward()

            if args.clip_grad_norm:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

            model.module.temperature.data.clamp_(-torch.log(torch.tensor(100.)), torch.log(torch.tensor(100.)))

            labels = torch.arange(args.rank*args.bs, (args.rank+1)*args.bs).to(args.rank)
            acc_text  = ((sim_text.argmax(1) == labels).float()).mean()
            acc_image = ((sim_image.argmax(1) == labels).float()).mean()
            acc       = (acc_text + acc_image) / 2

            reduced_loss = reduce_tensor(loss.data, args.world_size)
            losses.update(reduced_loss.item())

            reduced_acc = reduce_tensor(acc.data, args.world_size)
            accuracies.update(reduced_acc.item())

            if args.tb_profile:
                prof.step()

            bt = time.time() - tp
            bt = torch.tensor(bt).to(args.rank)
            bt = reduce_tensor(bt, args.world_size)
            batch_time.update(bt)

            if args.rank == 0:
                writer.add_scalars("1 loss/1 step", {"train": reduced_loss.item()}, step)
                writer.add_scalars("2 accuracy/1 step", {"train": reduced_acc.item()}, step)
                writer.add_scalars("3 temperature/1 step", {"train": model.module.temperature.data.item()}, step)
                writer.add_scalars("4 timings/1 step", {"dt": dt, "bt": bt}, step)
                if (step % args.save_interval_step == 0) and (step != 0):
                    path_save = os.path.join(args.path_model, f"{'_'.join(str(datetime.now()).split('.')[0].split(' '))}_step{step:08d}.pt")
                    torch.save(ddp_model.module.state_dict(), path_save)
                    logger.info(f"{datetime.now()} epoch: {epoch:>4} step: {step:>8} bt: {batch_time.avg:<10.3f}dt: {data_time.avg:<10.3f}{'train':<10} loss: {losses.avg:<10.3f} acc: {accuracies.avg:<10.3f}")

            if (step % args.save_interval_step == 0) and (step != 0):
                validate(args, model, dl_valid_id,  logger, writer, step, epoch, step_epoch="step", logid="valid-id")
                validate(args, model, dl_valid_ood, logger, writer, step, epoch, step_epoch="step", logid="valid-ood")

            step += 1

            tp = time.time()

        if args.tb_profile:
            prof.stop()   

        time_epoch_end = time.time()
        et = time_epoch_end - time_epoch_start
        et = torch.tensor(et).to(args.rank)
        epoch_time = reduce_tensor(et, args.world_size)

        if args.rank == 0:
            writer.add_scalars("1 loss/2 epoch", {"train": losses.avg}, epoch)
            writer.add_scalars("2 accuracy/2 epoch", {"train": accuracies.avg}, epoch)
            writer.add_scalars("4 timings/2 step", {"dt": data_time.avg, "bt": batch_time.avg}, epoch)
            writer.add_scalars("4 timings/3 epoch", {"et": epoch_time}, epoch)
            if epoch % args.save_interval_epoch == 0:
                path_save = os.path.join(args.path_model, f"{'_'.join(str(datetime.now()).split('.')[0].split(' '))}_epoch{epoch:03d}.pt")
                torch.save(ddp_model.module.state_dict(), path_save)
                logger.info(f"{datetime.now()} epoch: {epoch:>4} et: {epoch_time:<11.3f}bt: {batch_time.avg:<10.3f}dt: {data_time.avg:<10.3f}{'train':<10} loss: {losses.avg:<10.3f} acc: {accuracies.avg:<10.3f}")

        if epoch % args.save_interval_epoch == 0:
            validate(args, model, dl_valid_id,  logger, writer, step, epoch, step_epoch="epoch", logid="valid-id")
            validate(args, model, dl_valid_ood, logger, writer, step, epoch, step_epoch="epoch", logid="valid-ood")

        return model, optimizer, step

    logger.info(f"{datetime.now()} rank: {args.rank} start training")
    for epoch in range(args.epochs):
        ddp_model, optimizer, step = one_epoch(args, ddp_model, optimizer, dl_train, dl_valid, epoch, step)

    cleanup()
    logger.info(f"{datetime.now()} rank: {args.rank} ddp cleanup")


def run(func, world_size):
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def trainer(rank, world_size):

    # get args
    args = get_args()
    args.rank = rank
    args.world_size = world_size
    
    # setup paths
    args.path_log = os.path.join(args.path_results, args.id)
    args.path_tb = os.path.join(args.path_log,"tb")
    args.path_model = os.path.join(args.path_log,"model")
    os.makedirs(args.path_log, exist_ok=True)
    os.makedirs(args.path_tb, exist_ok=True)
    os.makedirs(args.path_model, exist_ok=True)

    # setup loggers
    # TO DO: Revisit file name when processes are more than 1 min apart.
    # (We want to log the setup from every rank to be able to debug all ranks.)
    fn_log = f"train_{'_'.join(str(datetime.now()).split('.')[0].split(' '))}.log"
    logger = create_logger(args.path_log, file_name=fn_log)
    logger.info(f"{datetime.now()} rank: {args.rank} start logging")
    if args.rank == 0:
        writer = SummaryWriter(log_dir=args.path_tb, flush_secs=2)

    if args.rank == 0:
        # print and log args
        for k in args.__dict__.keys():
            logger.info(f"{k:>20}: {args.__dict__[k]}")

    # data setup
    # TO DO: Add tokenizer and samplers!
    logger.info(f"{datetime.now()} rank: {args.rank} created samplers and tokenizers")

    logger.info(f"{datetime.now()} rank: {args.rank} data setup")

    ds_train = # TO DO: Add!
    logger.info(f"{datetime.now()} rank: {args.rank} created train dataset")

    dl_train = DataLoader(ds_train,
                          batch_size = args.bs,
                          shuffle = True if not(args.dryrun) else False,
                          num_workers = args.numw,
                          pin_memory = True,
                          drop_last = True)
    logger.info(f"{datetime.now()} rank: {args.rank} created train dataloader with length {len(dl_train)}")

    ds_valid = # TO DO: Add!
    logger.info(f"{datetime.now()} rank: {args.rank} created valid dataset")

    dl_valid = DataLoader(ds_valid,
                          batch_size = args.bs,
                          shuffle = False,
                          num_workers = args.numw,
                          pin_memory = True,
                          drop_last = False)
    logger.info(f"{datetime.now()} rank: {args.rank} created valid dataloader with length {len(dl_valid)}")

    torch.backends.cudnn.benchmark = True
    logger.info(f"{datetime.now()} rank: {args.rank} enabled cuDNN autotuner.")

    # model setup
    model = CLIP(
            dim_text = args.dim_text,
            dim_image = args.dim_image,
            dim_latent = args.dim_latent,
            num_text_tokens = args.num_text_tokens,
            text_enc_depth = args.text_enc_depth,
            text_seq_len = args.text_seq_len,
            text_heads = args.text_heads,
            num_visual_tokens = args.num_visual_tokens,
            visual_enc_depth = args.visual_enc_depth,
            visual_heads = args.visual_heads,
            visual_image_size = args.visual_image_size,
            visual_patch_size = args.visual_patch_size,
            channels = args.channels,
            use_all_token_embeds = args.use_all_token_embeds,
            downsample_image_embeds = args.downsample_image_embeds,
            decoupled_contrastive_learning = args.decoupled_contrastive_learning,
            extra_latent_projection = args.extra_latent_projection,
    )

    if args.path_weights:
        # TO DO: Check if we really need to load the weights on each rank for ddp.
        ckpt = torch.load(args.path_weights, map_location="cpu")
        model.load_state_dict(ckpt)
        logger.info(f"{datetime.now()} rank: {args.rank} reloaded model weights from {args.path_weights}")

    logger.info(f"{datetime.now()} rank: {args.rank} created CLIP model")

    # optimizer
    opt = AdamW(model.parameters(), lr = args.lr)
    logger.info(f"{datetime.now()} rank: {args.rank} created AdamW optimizer")

    # training
    if args.rank == 0: # only use tb writer in rank 0
        train_ddp(args, model=model, optimizer=opt,
                dl_train=dl_train, dl_valid=dl_valid,
                epochs=args.epochs, logger=logger, writer=writer)
    else:
        train_ddp(args, model=model, optimizer=opt,
                dl_train=dl_train, dl_valid=dl_valid,
                epochs=args.epochs, logger=logger)

    logger.info(f"{datetime.now()} rank: {args.rank} training finished")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"#gpus: {n_gpus}")
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
        run(trainer, n_gpus)
