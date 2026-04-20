import importlib.util
import os

# Get the directory where this __init__.py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try multiple possible locations for spatial_utils.so
possible_paths = [
    os.path.join(current_dir, "spatial_utils.so"),  # Same directory as __init__.py
    os.path.join(
        os.path.dirname(current_dir), "cpp_extensions", "spatial_utils.so"
    ),  # Project root
    os.path.join(
        os.path.dirname(os.path.dirname(current_dir)),
        "spatial_adapter",
        "cpp_extensions",
        "spatial_utils.so",
    ),  # Development install
]

spatial_utils = None
for spatial_utils_path in possible_paths:
    if os.path.exists(spatial_utils_path):
        try:
            # Load the C++ extension
            spec = importlib.util.spec_from_file_location(
                "spatial_utils", spatial_utils_path
            )
            spatial_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(spatial_utils)
            break
        except Exception as e:
            pass
            continue

if spatial_utils is None:
    raise ImportError(
        "Could not find or load spatial_utils.so. Tried paths: " + str(possible_paths)
    )

__all__ = [
    "smoothing_penalty_matrix",
    "interpolate_eigenfunction",
    "estimate_covariance",
    "fixed_rank_kriging",
]

# Import the functions from the compiled extension
smoothing_penalty_matrix = spatial_utils.smoothing_penalty_matrix
interpolate_eigenfunction = spatial_utils.interpolate_eigenfunction
estimate_covariance = spatial_utils.estimate_covariance
fixed_rank_kriging = spatial_utils.fixed_rank_kriging
