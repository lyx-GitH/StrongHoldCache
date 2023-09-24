import torch
from .stronghold_memory_manager import StrongHoldMemManager
from collections import OrderedDict

class FlowMemoryMananger(StrongHoldMemManager):
    def __init__(self, 
                 n_cpu_buffers: int, 
                 n_gpu_buffers: int, 
                 buffer_size: int, 
                 max_parallel_cuda_cpu_io: int, 
                 max_parallel_cpu_nvme_io: int, 
                 gpu_dtype: torch.dtype = torch.float32, 
                 cpu_dtype: torch.dtype = torch.float32, 
                 pre_g2c_func=None, 
                 pre_c2g_func=None
                 ) -> None:
        
        super().__init__(n_cpu_buffers, n_gpu_buffers, buffer_size, max_parallel_cuda_cpu_io, max_parallel_cpu_nvme_io, gpu_dtype, cpu_dtype, pre_g2c_func, pre_c2g_func)
        self.unit_cnt = 0
        self.staged_params = OrderedDict()
        self.staged_numel : int = 0
    
    def __flush_staged_params(self, replace_data_with_buffer_data : bool):
        self.init_nvme_block(self.unit_cnt, self.staged_params, replace_data_with_buffer_data=replace_data_with_buffer_data)
        self.unit_cnt += 1
        self.staged_numel = 0
        self.staged_params.clear()
        
    
    def stage_for_init(self, param, replace_data_with_buffer_data : bool = True):
        assert hasattr(param, "ds_id") and hasattr(param, "ds_numel"), "invalid deepspeed param"
        param_numel : int = param.ds_numel
        
        if self.staged_numel + param_numel > self.cpu_nvme_swapper.aligned_elements_per_buffer:
            self.__flush_staged_params()
        
        self.staged_params[param.ds_id] = param
        self.staged_numel += param_numel
        
    def get_units_for_params(self, params: list) -> list[int]:
        units = set()
        for param in params:
            units.add(self.cpu_nvme_swapper._get_param_unit_id(param.ds_id))
        return list(units)
    
    