import torch
from collections import OrderedDict
from .thread_safe_queue import ThreadSafeDeque
from .deepspeed_swapper import PartitionedParamStatus, ThreadSafeAsyncPartitionedParameterSwapper
from .io_handle import NvmeHandle
from .offload_utils import print_rank_0
from .offload_utils import ParamId

from memory_profiler import profile

class MockDsParam():
    '''
    This class simulates all methods and properties of ds_tensor used in APPS
    '''
    class MockDsTensor():
        def __init__(self, tensor: torch.Tensor):
            self.data = tensor
            self.ds_numel = tensor.numel()
            self.status = PartitionedParamStatus.AVAILABLE

    def __init__(self, tensor_id: str, tensor: torch.Tensor) -> None:
        self.ds_id = tensor_id
        self.ds_tensor = self.MockDsTensor(tensor)
        pass

class TensorLocation():
    '''
    A class to describe where the tensor is
    >>> tensor = APPS.get_buffer(buffer_id).narrow(0, offset, length).resize_(shape)
    '''

    def __init__(self, unit_id: str, offset: int, length: int, shape: torch.Size):
        self.unit_id = unit_id
        self.offset = offset
        self.length = length
        self.shape = shape
        self.is_del = False

    def __str__(self):
        return f'''
        unit_id:    {self.unit_id}
        offset:     {self.offset}
        length:     {self.length}
        shape:      {self.shape}
        is_del:     {self.is_del}
        '''

class StrongHoldParamSwapper(ThreadSafeAsyncPartitionedParameterSwapper):
    @ profile
    def __init__(self, ds_config, model_dtype):
        super().__init__(ds_config, model_dtype)
        self.param_locations = {}
        self.unit_param = {}
        self.fake_unit_params = {}
        self._init_handles(ds_config)
        print(self.aio_config.block_size)
        
    def _init_handles(self, ds_config):
        self.read_handles = ThreadSafeDeque()
        self.write_handles = ThreadSafeDeque()
        for hid in range(ds_config.aio_config.max_aio_handles):
            read_handle = self.aio_handle(self.aio_config.block_size, self.aio_config.queue_depth,
                                               self.aio_config.single_submit, self.aio_config.overlap_events,
                                               self.aio_config.thread_count)
            write_handle = self.aio_handle(self.aio_config.block_size, self.aio_config.queue_depth,
                                               self.aio_config.single_submit, self.aio_config.overlap_events,
                                               self.aio_config.thread_count)
            self.read_handles.deque.append(
                NvmeHandle(self, hid, read_handle, NvmeHandle.HandleType.READ)
                )
            self.write_handles.deque.append(
                NvmeHandle(self, hid, write_handle, NvmeHandle.HandleType.WRITE)
                )
    
    
    def apply_read_handle(self) -> NvmeHandle:
        '''
        Apply a read handle, would wait if no handle is available
        '''
        return self.read_handles.pop()
        
    def apply_write_handle(self) -> NvmeHandle:
        '''
        Apply a write handle, would wait if no handle is available
        '''
        return self.write_handles.pop()
    
    def revoke_handle(self, handle: NvmeHandle):
        '''
        returns this hanlde to the handle pool, for resue, this method assures that all IO is synced
        '''
        assert handle.swapper == self, "not this swapper"
        # handle.synchronize()
        ts_queue : ThreadSafeDeque = self.read_handles if handle.is_read_handle() else self.write_handles
        assert handle not in ts_queue.deque, f"handle {handle} already in queue"
        ts_queue.append(handle)
        
            

    def _get_buffer_slice(self, buffer_id: int):
        return self.buffers[buffer_id]

    def _all_marked_deleted(self, unit_id: str):
        params = self.unit_param[unit_id]
        return all(
            self.param_locations[param].is_del for param in params
        )

    def _get_param_unit_id(self, param_id: ParamId):
        assert param_id in self.param_locations, f"unknown param: {param_id}"
        return self.param_locations[param_id].unit_id

    def _is_in_cpu(self, param_id: ParamId):
        unit_id = self._get_param_unit_id(param_id)
        fake_param: MockDsParam = self.fake_unit_params[unit_id]
        return fake_param.ds_tensor.status == PartitionedParamStatus.AVAILABLE
    
    def _is_unit_in_cpu(self, unit_id: str):
        assert unit_id in self.fake_unit_params, f"unknown unit: {unit_id}"
        fake_param: MockDsParam = self.fake_unit_params[unit_id]
        return fake_param.ds_tensor.status == PartitionedParamStatus.AVAILABLE
        

    
    def _has_param(self, param_id: ParamId):
        return param_id in self.param_locations
    
    def _fill_buffer_block(self, buffer_id: int, unit_id: int, unit: OrderedDict, copy_data = True, set_as_invalid_block = False, replace_data_with_buffer_data = False):
        '''
        fill a unit to a certain buffer block, and returns the fake param containing this block
        '''
        # assert buffer_id in self.available_buffer_ids, f"buffer {buffer_id} has been occupied"
        unit_numel_cnt = 0
        # remove the swapping buffer from the APPS lists
        buffer_slice = self._get_buffer_slice(buffer_id)
        self.available_buffer_ids.remove(buffer_id)
        params_in_this_unit = []

        # copy each param in a unit into a single buffer
        for param_id, param in unit.items():
            buffer_slice = self._get_buffer_slice(buffer_id)
            print_rank_0(
                f'loading {param_id} with shape {param.shape} to buffer')
            param_location = TensorLocation(
                unit_id, unit_numel_cnt, param.numel(), param.shape)
            
            # print(f"!!!!!!!!!!!!!! {param.shape}")
            self.param_locations[param_id] = param_location
            # print(f"is_require_grad : {buffer_slice}")
            
            swapping_area = buffer_slice.narrow(
                0, unit_numel_cnt, param.numel()
                ).resize_(param_location.shape)
            
            if copy_data:
                swapping_area.copy_(param)
            else:
                swapping_area.zero_()
            
            if replace_data_with_buffer_data:
                param.data = swapping_area
                
            unit_numel_cnt += param.numel()
            # print_rank_0(
            #     f'copy {param_id} with shape {param.shape} to {self.param_locations[param_id]}')
            params_in_this_unit.append(param_id)

            assert unit_numel_cnt <= self.aligned_elements_per_buffer, "not enough space for this layer"

        
        fake_param = MockDsParam(unit_id, buffer_slice.narrow(0, 0, unit_numel_cnt))
        
        if set_as_invalid_block:
            fake_param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE
        # set bookeeping info
        self.unit_param[unit_id] = params_in_this_unit
        self.fake_unit_params[unit_id] = fake_param
        
        return fake_param
    
    def _get_tensor_from_buffer(self, location: TensorLocation):
        unit_id = location.unit_id
        buf_id = self.param_id_to_buffer_id[unit_id]
        return self._get_buffer_slice(buf_id).narrow(
            0, location.offset, location.length
        ).view(location.shape)
    
    def _set_tensor_to_buffer(self, location: TensorLocation, tensor: torch.Tensor, async_op = False):
        unit_id = location.unit_id
        buf_id = self.param_id_to_buffer_id[unit_id]
        swapping_area = self._get_buffer_slice(buf_id).narrow(
            0, location.offset, location.length
        ).view(location.shape)
        swapping_area.copy_(tensor, non_blocking = async_op)
        
    
    def _save_unit_info(self, buffer_id: int, unit_id: str, fake_param: MockDsParam):
        self.param_id_to_buffer_id[unit_id] = buffer_id
        self.param_id_to_numel[unit_id] = fake_param.ds_tensor.ds_numel
        self.param_id_to_swap_buffer[unit_id] = fake_param.ds_tensor.data
    
    def _drop_unit(self, unit_id: str):
        self.remove_partition_and_release_buffers(
            [self.fake_unit_params[unit_id]]
        )
        
    @torch.no_grad()
    def init_nvme_block(self, unit_id: str, unit: OrderedDict, copy_data = True, replace_data_with_buffer_data = False):
        '''
        One unit (layer) in SH points to one buffer in swapper.
        must assure that the CPU memory is enough for one layer.
        This method always operates the first layer, put one layer into the cpu
        then flush it
        if copy_data is set to true, the data would be copied, otherwise, just alloc
        a *empty* buffer for this unit
        '''
        assert self.available_swap_in_buffers() > 0, "unable to alloc buffer"
        assert unit_id not in self.fake_unit_params, "this unit has been initialized!"
        print_rank_0(f"init nvme block: {unit_id}")

        buf_id = self.available_buffer_ids.deque[0]
        fake_param = self._fill_buffer_block(buf_id, unit_id, unit, copy_data=copy_data, replace_data_with_buffer_data=replace_data_with_buffer_data) # set this block to invalid
        # fake the original bookeeping info
        self._save_unit_info(buf_id, unit_id, fake_param)
        write_handle = self.apply_write_handle()
        self.swap_out_and_release(write_handle, [fake_param])
        # 
        write_handle.synchronize()
        assert not self._is_in_cpu(list(unit.keys())[0])
        
    @torch.no_grad()
    def init_cpu_block(self, unit_id: str, unit: OrderedDict):
        '''
        put one block directly into the cpu, and bookkeeping this block's info
        '''
        assert self.available_swap_in_buffers() > 0, "unable to alloc buffer"
        assert unit_id not in self.unit_param, "this unit has been initialized"
        target_buffer = self.available_buffer_ids.pop()
        fake_param = self._fill_buffer_block(target_buffer, unit_id, unit)
        self._save_unit_info(target_buffer, unit_id, fake_param)
        return fake_param
    
    def _is_inflight(self, handle: NvmeHandle, param_id: str):
        unit_id = self._get_param_unit_id(param_id)
        fake_param : MockDsParam = self.fake_unit_params[unit_id]
        return fake_param in handle.inflight_params
    
    def _is_inflight_write(self, handle: NvmeHandle, param_id: str):
        unit_id = self._get_param_unit_id(param_id)
        fake_param : MockDsParam = self.fake_unit_params[unit_id]
        return fake_param in handle.swap_out_params
        
    
    def async_load(self, handle: NvmeHandle, param_id: str):
        '''
        async load that param into the nvme
        '''
        assert handle.is_read_handle(), "requires a read handle"
        if not self._is_in_cpu(param_id) and not self._is_inflight(handle, param_id): 
            unit_id = self._get_param_unit_id(param_id)
            print_rank_0(f"[SWAPPER]: begin swap in {unit_id}")
            self.swap_in(handle, unit_id, async_op=True)
            
    
    def async_flush(self, handle: NvmeHandle, param_id: str):
        '''
        async save that param to the nvme and DROP that block
        '''
        assert not handle.is_read_handle(), "requires a write handle"
        if self._is_in_cpu(param_id) and not self._is_inflight_write(handle, param_id):
            fake_param = self.fake_unit_params[self._get_param_unit_id(param_id)]
            print_rank_0(f"[SWAPPER]: begin swap out {self._get_param_unit_id(param_id)} : {param_id}")
            self.swap_out_and_release(handle, [fake_param], async_op=True, force_buffer_release=True)
            
        

    def get(self, param_id: str, handle : NvmeHandle = None):
        '''
        get the param from the cpu buffer, if this buffer is not in the cpu, swap it in
        this is always synced, if handle is set to None, just fecth from the cpu
        '''
        assert param_id in self.param_locations, f"unrecognized param: {param_id}"
        location: TensorLocation = self.param_locations[param_id]
        if location.unit_id in self.param_id_to_buffer_id:
            return self._get_tensor_from_buffer(location)
        else:
            assert handle is not None, f"param {param_id} not in CPU"
            self.swap_in(handle, location.unit_id)
            return self.get(param_id, handle)
        
    
    def put(self, param_id: str, tensor: torch.Tensor, non_cuda_blocking = False, handle: NvmeHandle = None):
        '''
        put a tensor to the cache system. if `non_cuda_blocking` is True,
        the cuda-cpu writing process would be async, but nvme IO is always synced
        '''
        assert self._has_param(param_id), f"not recognized param: {param_id}"
        location = self.param_locations[param_id]
        if not self._is_in_cpu(param_id):
            assert handle is not None, f"param {param_id} not in CPU"
            self.swap_in(handle, param_id, async_op=False)
        self._set_tensor_to_buffer(location, tensor, async_op=non_cuda_blocking)
        
        

    def swap_in(self, handle: NvmeHandle, param_id: str, async_op=False):
        '''
        swap the unit of the param to the cpu
        '''
        if not self._is_unit_in_cpu(param_id):
            fake_param = self.fake_unit_params[param_id]
            super().swap_in(handle, [fake_param], async_op=async_op)

    def drop(self, param_id: str, async_op = False, force = False):
        '''
        drop a certain param,
        if `force` is set to True, drop this unit, otherwise, this unit would be dropped
        once all the params are dropped
        '''
        if self._is_in_cpu(param_id):
            self.param_locations[param_id].is_del = True
            unit_id = self._get_param_unit_id(param_id)
            if self._all_marked_deleted(unit_id):
                self._drop_unit(unit_id)
        pass
    
