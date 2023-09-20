from copy import deepcopy
ds_config = {
    "zero_config": {
        "stage": 0,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/shared_ssd_storage/yuxuanliu/STRONGHOLD/",
            "pin_memory": True,
            "buffer_count": 4,
            "buffer_size": 15735296,
            "max_in_cpu": 1e9
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        # "reduce_bucket_size": model_hidden_size * model_hidden_size,
        # "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        # "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "aio_config": {
        "block_size": 131072,
        "queue_depth": 16,
        "thread_count": 1,
        "single_submit": True,
        "overlap_events": True,
        "max_aio_handles": 24
    },
}


class DsConfig:
    '''
    Change json literal object to a python object
    '''

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DsConfig(value))
            else:
                setattr(self, key, value)

    def __repr__(self, indent=0):
        lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, DsConfig):
                lines.append(f"{' ' * indent}{key}:")
                lines.append(value.__repr__(indent=indent+2))
            else:
                lines.append(f"{' ' * indent}{key}: {value}")
        return "".join(lines)


def get_swapper_config(block_size = None, num_blocks = None, num_aio_handles = None) -> DsConfig:
    
    def _valid_num(value):
        return type(value) in (int, float) and value >= 1
    _ds_config = deepcopy(ds_config)
    if block_size is not None:
        assert _valid_num(block_size), f"invalid value {block_size} for block_size"
        _ds_config["zero_config"]["offload_param"]["buffer_size"] = int(block_size)
    
    if num_blocks is not None:
        assert _valid_num(block_size), f"invalid value {num_blocks} for num_blocks"
        _ds_config["zero_config"]["offload_param"]["buffer_count"] = int(num_blocks)
    
    if num_aio_handles is not None:
        assert _valid_num(num_aio_handles), f"invalid value {num_aio_handles} for num_aio_handles"
        _ds_config["aio_config"]["max_aio_handles"] = int(num_aio_handles)
        
    # print(ds_config)
    ds_config_tup = DsConfig(_ds_config)
    return ds_config_tup
