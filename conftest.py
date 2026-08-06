"""
conftest.py — pytest configuration for TruthLens.
Provides fixtures and marks tests that require torch.
"""
import pytest

def pytest_collection_modifyitems(config, items):
    """Skip tests that require torch if torch is not installed."""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        skip_torch = pytest.mark.skip(reason="torch not installed")
        torch_test_names = [
            "test_train_transform_output_shape",
            "test_val_transform_output_shape",
            "test_test_transform_output_shape",
            "test_normalisation_applied",
            "test_from_pil_returns_tensor",
            "test_from_numpy_returns_tensor",
            "test_unsupported_type_raises",
            "test_return_numpy_when_flagged",
            "test_default_config_valid",
            "test_config_serialisation",
            "test_overlay_output_shape",
            "test_overlay_value_range",
            "test_build_result_structure",
            "test_build_result_real_class",
            "test_seed_everything",
            "test_get_device_returns_device",
            "test_logger_created",
        ]
        for item in items:
            if item.name in torch_test_names:
                item.add_marker(skip_torch)
