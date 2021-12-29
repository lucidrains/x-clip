import torch

from .distributed_backend import DistributedBackend


class HorovodBackend(DistributedBackend):
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
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        self.backend_module.init_process_group("nccl", rank=self._get_local_rank, world_size=self._get_world_size)

        if torch.cuda.is_available():
            torch.cuda.set_device(self._get_local_rank())

        assert self.backend_module.is_initialized(), "PyTorch DDP backend is initialized."

    def _get_world_size(self):
        return self.backend_module.get_world_size()

    def _get_rank(self):
        return self.backend_module.get_rank()

    def _get_local_rank(self):
        #return self.backend_module.local_rank()
        # TO DO: Check how local vs global ranks is handled with PyTorch DDP. In the meantime wie return self._get_rank
        return self._get_rank() # == return self.backend_module.get_rank()

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
        model.to(self._get_local_rank())
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self._get_local_rank()])

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
        averaged = self.backend_module.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, async_op=False) / self._get_world_size()
        return averaged
