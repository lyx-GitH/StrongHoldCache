from enum import Enum
from abc import ABC, abstractmethod
import torch

class PartitionedParamStatus(Enum):
    # Partitioned parameters are present and ready for use
    AVAILABLE = 1

    # partitioned params are in some non-memory device
    NOT_AVAILABLE = 2

    # partitioned params are being read from some non-memory device.
    INFLIGHT = 3



class IOHandle(ABC):
    def __init__(self, hid: int) -> None:
        super().__init__()
        self.hid = hid
    @abstractmethod
    def synchronize():
        pass
    
    def __str__(self) -> str:
        return f"{type(self)}-{self.hid}"
    


class CudaHandle(IOHandle):
    def __init__(self, hid: int, cuda_swapper,  *args, **kwargs) -> None:
        super().__init__(hid)
        self.cuda_swapper = cuda_swapper
        self.__stream = torch.cuda.Stream(*args, **kwargs)
        self.running = False
        
        
    def synchronize(self):
        '''
        Synchromize the stream and RETURN THIS HANDLE to the swapeer
        '''
        if self.running:
            self.running = False
            # self.__stream.synchronize()
            torch.cuda.current_stream().wait_stream(self.__stream)
            self.cuda_swapper._revoke_handle(self)
    
    def get_stream(self) -> torch.cuda.Stream:
        assert not self.running, f"{self} is not sychronized!"
        self.running = True
        return self.__stream
    
    
    
    
    

class NvmeHandle(IOHandle):
    '''
    This class manages all IOs related to the SSD.
    '''
    class HandleType(Enum):
        READ = 0
        WRITE = 1
        
    def __init__(self, swapper, hid, handle, htype: HandleType) -> None:
        super().__init__(hid)
        self.swapper = swapper
        self.type = htype
        self.handle = handle
        self.pending_reads = 0
        self.pending_writes = 0
        self.inflight_numel = 0
        self.inflight_swap_in_buffers = []
        self.inflight_params = []
        
        self.available_numel = 0
        self.available_params = set()
        
        self.swap_out_params = []
    
    def is_read_handle(self):
        return self.HandleType.READ == self.type
    
    def __str__(self) -> str:
        htype = 'read' if self.is_read_handle() else 'write'
        return f"{htype}_handle_{self.hid}"

    def synchronize(self):
        '''
        Synchronize all async IOs on the ssd, and RETURN this to the swapper
        '''
        if self.is_read_handle():
            self._synchronize_reads()
        else:
            self._synchronize_writes()
        
        self.swapper.revoke_handle(self)
        
            
    def async_pwrite(self, *args, **kwargs):
        return self.handle.async_pwrite(*args, **kwargs)
    
    def async_pread(self, *args, **kwargs):
        return self.handle.async_pread(*args, **kwargs)
        
    def _synchronize_reads(self):
        if self.pending_reads == 0:
            return

        assert self.pending_reads == self.handle.wait()

        self.pending_reads = 0

        for param, swap_in_buffer in zip(self.inflight_params, self.inflight_swap_in_buffers):
            param_id = param.ds_id
            compute_buffer = swap_in_buffer.narrow(
                0, 0, self.swapper.param_id_to_numel[param_id])
            param.ds_tensor.data = compute_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE

        self.available_params.update(
            [param.ds_id for param in self.inflight_params])
        self.available_numel += self.inflight_numel
        
        for p in self.inflight_params:
            print(f"[SWAPPER] {self} finish reading {p.ds_id}")

        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0
    
    
    def _synchronize_writes(self):
        if self.pending_writes == 0:
            return
        assert self.pending_writes == self.handle.wait()
        self.pending_writes = 0
        self.swapper.remove_partition_and_release_buffers(self.swap_out_params)
        for p in self.swap_out_params:
            print(f"[SWAPPER] {self} finish write {p.ds_id}")
        self.swap_out_params = []
         
    pass

