import os
import torch

from .distributed_backend import DistributedBackend


class PyTorchDDPBackend(DistributedBackend):
    """Distributed backend using Horovod."""

    BACKEND_MODULE_NAME = 'torch.distributed'
    BACKEND_NAME = 'PyTorch DDP'

    def wrap_arg_parser(self, parser):
        return parser

    def check_batch_size(self, batch_size):
        # PyTorch DDP uses the local batch size to determine the effective
        # batch size.
        pass

    def _initialize(self):

        assert self.backend_module.is_available(), "PyTorch DDP backend is not available."

        # get environment variable
        self.world_size = int(os.getenv("SLURM_NTASKS"))
        self.rank       = int(os.getenv("SLURM_PROCID"))
        self.local_rank = int(os.getenv("SLURM_LOCALID"))

        # print slurm setup for debugging
        print(f"rank: {self.rank} MASTER_ADDR: {os.getenv('MASTER_ADDR')}")
        print(f"rank: {self.rank} SLURM_NTASKS = world_size: {os.getenv('SLURM_NTASKS')}")
        print(f"rank: {self.rank} SLURM_PROCID = rank: {os.getenv('SLURM_PROCID')}")
        print(f"rank: {self.rank} SLURM_LOCALID = local_rank: {os.getenv('SLURM_LOCALID')}")

        # initialize the process group
        self.backend_module.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)

        # TO DO: Check if we can remove that from the training loop and put it here?
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            # TO DO: Check if still needed with latest PyTorch version.
            # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
            #torch.cuda.empty_cache()

        assert self.backend_module.is_initialized(), "PyTorch DDP backend is not initialized."

    def _get_world_size(self):
        return self.world_size

    def _get_rank(self):
        return self.rank

    def _get_local_rank(self):
        return self.local_rank

    def _local_barrier(self):
        self.backend_module.barrier()

    def _distribute(
            self,
            _args=None,
            model=None,
            optimizer=None,
            _model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            find_unused_parameters=True, # TO DO: Check why this is needed?
            **_kwargs,
    ):
        # TO DO: Horovod setup uses self.ROOT_RANK, investigate why and if we need that setup here.
        model.to(self.local_rank)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model,
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused_parameters)
        return (ddp_model, optimizer, training_data, lr_scheduler)

    def _average_all(self, tensor):
        reduced_tensor = tensor.clone()
        torch.distributed.all_reduce(reduced_tensor, op=torch.distributed.ReduceOp.SUM, async_op=False) # Reduce op is average by default
        reduced_tensor /= self.get_world_size()
        return reduced_tensor
