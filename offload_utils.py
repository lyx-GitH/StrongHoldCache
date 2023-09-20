import torch
from dataclasses import dataclass

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


@dataclass
class ParamId:
    layer_id : int
    param_id: str
    
    def __hash__(self) -> int:
        return self.layer_id.__hash__() + self.param_id.__hash__()