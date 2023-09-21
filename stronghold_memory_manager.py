import torch
from collections import OrderedDict
from .thread_safe_queue import ThreadSafeDeque
from .io_handle import PartitionedParamStatus, NvmeHandle, CudaHandle
from .stronghold_param_swapper import MockDsParam, StrongHoldParamSwapper, TensorLocation
from .swapper_config import get_swapper_config
from .offload_utils import ParamId


class CudaParamSwapper():
    def __init__(self,
                 dtype: torch.dtype,
                 n_cuda_streams: int,
                 n_cuda_buffers: int,
                 cuda_buf_size: int,
                 c2g_f ,
                 g2c_f) -> None:

        self.dtype = dtype
        self.n_cuda_streams = n_cuda_streams
        self.n_cuda_buffers = n_cuda_buffers
        self.cuda_buf_size = cuda_buf_size
        self.c2g_f = c2g_f
        self.g2c_f = g2c_f
        self.unit_id_to_cuda_id = {}
        self._init_cuda_buffers()
        self._init_cuda_handles()
        

    def _apply_handle(self) -> CudaHandle:
        return self.cuda_streams.pop()

    def _revoke_handle(self, cuda_handle: CudaHandle):
        '''
        sync this stream and returns to the manager
        '''
        assert cuda_handle not in self.cuda_streams.deque, f"handle {cuda_handle} already in queue"
        # cuda_handle.synchronize()
        self.cuda_streams.append(cuda_handle)

    def _init_cuda_buffers(self):
        buf_device = torch.device(torch.cuda.current_device())
        self.cuda_available_buffer_ids = ThreadSafeDeque(
            [i for i in range(self.n_cuda_buffers)])
        self.cuda_buffers = [torch.empty(
            self.cuda_buf_size, dtype=self.dtype, device=buf_device) for _ in range(self.n_cuda_buffers)]

    def _init_cuda_handles(self):
        self.cuda_streams = ThreadSafeDeque(
            [CudaHandle(i, self) for i in range(self.n_cuda_streams)])

    def _cuda_swap_in_tensor(self, cuda_buf_id: int, cpu_buf: torch.Tensor, non_blocking: bool = True):

        if self.c2g_f:
            cpu_buf = self.c2g_f(cpu_buf)

        cuda_buf = self.cuda_buffers[cuda_buf_id]
        return cuda_buf.copy_(
            cpu_buf,
            non_blocking=non_blocking
        )

    def _cuda_swap_out_tensor(self, cuda_buf_id: int, cpu_buf: torch.Tensor, non_blocking: bool = True):

        cuda_buf = self.cuda_buffers[cuda_buf_id] if not self.g2c_f else self.g2c_f(
            self.cuda_buffers[cuda_buf_id])

        return cpu_buf.copy_(
            cuda_buf,
            non_blocking=non_blocking
        )

    def _alloc_gpu_buf(self, unit_id) -> int:
        assert unit_id not in self.unit_id_to_cuda_id, f"unit_id {unit_id} already in cuda"
        cuda_buf_id = self.cuda_available_buffer_ids.pop()
        self.unit_id_to_cuda_id[unit_id] = cuda_buf_id
        return cuda_buf_id

    def _free_gpu_buf(self, unit_id) -> None:
        assert unit_id in self.unit_id_to_cuda_id, f"unit_id {unit_id} not in cuda"
        cuda_buf_id: int = self.unit_id_to_cuda_id[unit_id]
        del self.unit_id_to_cuda_id[unit_id]
        self.cuda_available_buffer_ids.append(cuda_buf_id)

    def _is_unit_in_cuda(self, unit_id) -> None:
        return unit_id in self.unit_id_to_cuda_id
    
    def get(self, location: TensorLocation):
        assert location.unit_id in self.unit_id_to_cuda_id, f"{location.unit_id=} not in cuda"
        cuda_buf_id = self.unit_id_to_cuda_id[location.unit_id]
        buffer : torch.Tensor = self.cuda_buffers[cuda_buf_id]
        return buffer.narrow(0, location.offset, location.length).view(location.shape)


class StrongHoldMemManager():
    '''
    A class for memory management

    The symbols are:

    c : cpu

    g: gpu

    n: nvme

    '''

    def __init__(self,
                 n_cpu_buffers: int,
                 n_gpu_buffers: int,
                 buffer_size: int,

                 max_parallel_cuda_cpu_io: int,
                 max_parallel_cpu_nvme_io: int,

                 gpu_dtype: torch.dtype = torch.float32,
                 cpu_dtype: torch.dtype = torch.float32,

                 pre_g2c_func = None,
                 pre_c2g_func = None
                 ) -> None:

        self.cpu_nvme_swapper = StrongHoldParamSwapper(
            ds_config=get_swapper_config(
                block_size=buffer_size,
                num_blocks=n_cpu_buffers,
                num_aio_handles=max_parallel_cpu_nvme_io
            ),
            model_dtype=cpu_dtype
        )

        self.cuda_cpu_swapper = CudaParamSwapper(
            dtype=gpu_dtype,
            n_cuda_streams=max_parallel_cuda_cpu_io,
            n_cuda_buffers=n_gpu_buffers,
            cuda_buf_size=self.cpu_nvme_swapper.aligned_elements_per_buffer,
            c2g_f=pre_c2g_func,
            g2c_f=pre_g2c_func
        )
        
        self.available_buffer_ids = {
            'cuda' : self.cuda_cpu_swapper.cuda_available_buffer_ids,
            'cpu' : self.cpu_nvme_swapper.available_buffer_ids
        }
        
        self.available_handles = {
            'cuda' : self.cuda_cpu_swapper.cuda_streams,
            'nvme_read' : self.cpu_nvme_swapper.read_handles,
            'nvme_write' : self.cpu_nvme_swapper.write_handles
        }

    def avail_cpu_buffers(self) -> int:
        return len(self.cpu_nvme_swapper.available_buffer_ids)

    def avail_gpu_buffers(self) -> int:
        return len(self.cuda_cpu_swapper.cuda_available_buffer_ids)

    def init_nvme_block(self, unit_id: str, unit: OrderedDict, copy_data: bool = True, replace_data_with_buffer_data: bool = False) -> None:
        self.cpu_nvme_swapper.init_nvme_block(
            unit_id, unit, copy_data, replace_data_with_buffer_data)

    def init_cpu_block(self, unit_id: str, unit: OrderedDict, copy_data: bool = True, replace_data_with_buffer_data: bool = False) -> None:
        self.init_nvme_block(self, unit_id, unit, copy_data=copy_data,
                             replace_data_with_buffer_data=replace_data_with_buffer_data)
        self.n2c([unit_id])

    def init_gpu_block(self, unit_id: str, unit: OrderedDict, copy_data: bool = True, replace_data_with_buffer_data: bool = False, only_in_cuda=False) -> None:
        self.init_cpu_block(self, unit_id, unit, copy_data=copy_data,
                            replace_data_with_buffer_data=replace_data_with_buffer_data)
        self.c2g([unit_id])
        if only_in_cuda:
            self.c2n([unit_id], writeback=False)

    def __cuda_swap_in(self, units):
        for unit_id in units:
            if self.cuda_cpu_swapper._is_unit_in_cuda(unit_id):
                continue
            assert self.cpu_nvme_swapper._is_unit_in_cpu(
                unit_id), f"{unit_id} not in cpu"
            cpu_buf_id = self.cpu_nvme_swapper.param_id_to_buffer_id[unit_id]
            cpu_buf = self.cpu_nvme_swapper._get_buffer_slice(cpu_buf_id)
            cuda_buf_id = self.cuda_cpu_swapper._alloc_gpu_buf(unit_id)
            self.cuda_cpu_swapper._cuda_swap_in_tensor(cuda_buf_id, cpu_buf)

    def __cuda_swap_out(self, units, writeback: bool):
        for unit_id in units:
            if self.cuda_cpu_swapper._is_unit_in_cuda(unit_id):
                continue
            assert self.cpu_nvme_swapper._is_unit_in_cpu(
                unit_id), f"{unit_id} not in cpu"
            cpu_buf_id = self.cpu_nvme_swapper.param_id_to_buffer_id[unit_id]
            cpu_buf = self.cpu_nvme_swapper._get_buffer_slice(cpu_buf_id)
            cuda_buf_id = self.cuda_cpu_swapper.unit_id_to_cuda_id[unit_id]
            if writeback:
                self.cuda_cpu_swapper._cuda_swap_out_tensor(
                    cuda_buf_id, cpu_buf)
            self.cuda_cpu_swapper._free_gpu_buf(unit_id)

    def async_c2g(self, units: list) -> CudaHandle | None:
        '''
        cpu -> gpu
        '''
        if all(self.cuda_cpu_swapper._is_unit_in_cuda(unit_id) for unit_id in units):
            return None
        cuda_handle = self.cuda_cpu_swapper._apply_handle()
        with torch.cuda.stream(cuda_handle.get_stream()):
            self.__cuda_swap_in(units)
            # self.cuda_cpu_swapper._revoke_handle(stream_)
        return cuda_handle

    def async_g2c(self, units: list, writeback=True) -> CudaHandle | None:
        '''
        gpu -> cpu
        '''
        if all(not self.cuda_cpu_swapper._is_unit_in_cuda(unit_id) for unit_id in units):
            return None

        self.n2c(units)  # assure all data is in cpu
        cuda_handle = self.cuda_cpu_swapper._apply_handle()
        with torch.cuda.stream(cuda_handle.get_stream()):
            self.__cuda_swap_out(units, writeback=writeback)
        return cuda_handle


    def async_c2n(self, units: list, writeback=True) -> NvmeHandle | None:

        if writeback:
            handle = self.cpu_nvme_swapper.apply_write_handle()
            self.cpu_nvme_swapper.swap_out_and_release(
                handle, units, async_op=True, force_buffer_release=True)
            # self.cpu_nvme_swapper.revoke_handle(handle)
            return handle
        else:
            self.cpu_nvme_swapper.remove_partition_and_release_buffers(units)
            return None

    def async_n2c(self, units: list) -> NvmeHandle | None:
        handle = self.cpu_nvme_swapper.apply_read_handle()
        for unit_id in units:
            self.cpu_nvme_swapper.swap_in(handle, unit_id, async_op=True)

        # self.cpu_nvme_swapper.revoke_handle(handle)
        return handle
    
    def c2g(self, units: list) -> None:
        handle = self.async_c2g(units)
        if handle:
            handle.synchronize()
    
    def g2c(self, units: list) -> None:
        handle = self.async_g2c(units)
        if handle:
            handle.synchronize()
    
    def n2c(self, units: list) -> None:
        handle = self.async_n2c(units)
        if handle:
            handle.synchronize()
    
    def c2n(self, units: list) -> None:
        handle = self.async_n2c(units)
        if handle:
            handle.synchronize()
    
    
    def fetch_from_cpu(self, unit_id, unit: OrderedDict):
        assert self.cpu_nvme_swapper._is_unit_in_cpu(unit_id)
        for param_id, param in unit.items():
            param.data = self.cpu_nvme_swapper.get(param_id)
            
    
    def fetch_from_cuda(self, unit_id, unit: OrderedDict):
        assert self.cuda_cpu_swapper._is_unit_in_cuda(unit_id)
        for param_id, param in unit.items():
            location : TensorLocation = self.cpu_nvme_swapper.param_locations[param_id]
            param.data = self.cuda_cpu_swapper.get(location)
    
    def get_param_cuda(self, param_id : ParamId):
        assert param_id in self.cpu_nvme_swapper.param_locations, f"unrecognized param {param_id}"
        location = self.cpu_nvme_swapper.param_locations[param_id]
        return self.cuda_cpu_swapper.get(location)
    
    
    def get_param_cpu(self, param_id : ParamId):
        return self.cpu_nvme_swapper.get(param_id)
    
    
        
    
        
