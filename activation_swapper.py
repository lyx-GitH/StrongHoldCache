import torch
from .stronghold_param_swapper import StrongHoldParamSwapper

class ActivationLocation:
    def __init__(self, length: int, units: int) -> None:
        self.length = length
        self.units = units
        pass
    

class ActivationSwapper(StrongHoldParamSwapper):
    '''
    This swapper is designed for storing activations
    one activation can now be stored in multiple blocks
    '''
    
    def __init__(self, ds_config, model_dtype):
        super().__init__(ds_config, model_dtype)
        self.activation_unit_ids = {}
    
    def _units_needed(self, activation: torch.Tensor):
        block_size = self.aligned_elements_per_buffer
        if activation.numel() % block_size == 0:
            return activation.numel() // block_size
        else:
            return activation.numel() // block_size + 1
    
    def _get_unit_key(self, act_key: str, unit_index: int):
        return act_key + str(unit_index)
        
    def init_activation(self, act_key:str, activation: torch.Tensor):
        '''
        Virtually alloc a activation
        '''
        assert act_key not in self.activation_unit_ids, f"already init: {act_key}"
        
        location = ActivationLocation(0, 0)
        self.activation_unit_ids[act_key] = location
    
    def _get_tensor_slice(self, tensor: torch.Tensor):
        pass
        
        
    
    def put(act_key: str, activation: torch.Tensor, cuda_stream: torch.cuda.Stream):
        
        
        pass
    
    def load_cuda(act_key: str, activation: torch.Tensor, cuda_stream: torch.cuda.Stream):
        
        pass
        
    
    