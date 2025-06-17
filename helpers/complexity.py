import torch
import torchinfo
import copy

MAX_MACS = 30_000_000
MAX_PARAMS_MEMORY = 128_000


def get_torch_macs_memory(model, input_size):
    if isinstance(input_size, torch.Size):
        input_size = tuple(input_size)

    if isinstance(input_size, torch.Tensor):
        input_size = tuple(input_size.size())

    # copy model and convert to full precision,
    # as torchinfo requires full precision to calculate macs
    model_for_profile = copy.deepcopy(model).float()

    model_profile = torchinfo.summary(model_for_profile, input_size=input_size, verbose=0)
    return model_profile.total_mult_adds, get_model_size_bytes(model)


def get_model_size_bytes(model: torch.nn.Module) -> int:
    """
    Calculate total model size in bytes, accounting for mixed parameter dtypes.

    Args:
        model: torch.nn.Module

    Returns:
        Total size in bytes (int)
    """
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
        torch.qint8: 1,
        torch.quint8: 1,
    }

    total_bytes = 0
    for param in model.parameters():
        dtype = param.dtype
        num_elements = param.numel()
        bytes_per_param = dtype_to_bytes.get(dtype)

        if bytes_per_param is None:
            raise ValueError(f"Unsupported dtype: {dtype}. Please implement yourself.")

        total_bytes += num_elements * bytes_per_param

    return total_bytes
