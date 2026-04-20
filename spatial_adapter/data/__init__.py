from .generators import generate_combined_synthetic_data, generate_time_synthetic_data
from .synthetic_binary import get_synthetic_binary_dataloader_and_val
from .synthetic_patch_binary import get_synthetic_patch_dataloader_and_val

# gwhd requires torchvision which is an optional dependency;
# import lazily so the core package doesn't break in envs without it.
try:
    from .gwhd import get_gwhd_dataloader_and_val
except Exception:
    get_gwhd_dataloader_and_val = None

__all__ = [
    "generate_combined_synthetic_data",
    "generate_time_synthetic_data",
    "get_gwhd_dataloader_and_val",
    "get_synthetic_binary_dataloader_and_val",
    "get_synthetic_patch_dataloader_and_val",
]
