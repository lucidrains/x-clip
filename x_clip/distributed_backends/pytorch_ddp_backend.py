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
        self.rank = int(os.getenv("SLURM_PROCID"))
        self.local_rank = int(os.getenv("SLURM_LOCALID"))
        print(f"init: self.world_size: {self.world_size}, self.rank: {self.rank}, self.local_rank: {self.local_rank}")
        #self.device = torch.device('cuda', self.local_rank)
        #self.device = torch.device(self.local_rank)
        #print(f"init: self.world_size: {self.world_size}, self.rank: {self.rank}, self.local_rank: {self.local_rank}, self.device: {self.device}")

        # initialize the process group
        self.backend_module.init_process_group(backend="nccl", rank=self.local_rank, world_size=self.world_size)

        # TO DO: Check if we can remove that from the training loop and keep it here?
        #if torch.cuda.is_available():
        #    torch.cuda.set_device(self.device)
        #    torch.cuda.set_device(self.local_rank)

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
            **_kwargs,
    ):
        # TO DO: Horovod setup uses self.ROOT_RANK, investigate why and if we need that setup here.
        #model.to(self.device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])

        # Based on: https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814/8
        # Remove parameters and add new parameter group after model is wrapped in DDP.
        optimizer.param_group.clear() # optimizer.param_group = []
        optimizer.state.clear() # optimizer.state = defaultdict(dict)
        optimizer.add_param_group(
                {'params' : [p for p in ddp_model.parameters()]}
                )

        return (ddp_model, optimizer, training_data, lr_scheduler)

    def _average_all(self, tensor):
        # Reduce op is average by default
        averaged = self.backend_module.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, async_op=False) / self.get_world_size()
        return averaged
