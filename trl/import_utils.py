import sys
from importlib.util import find_spec


if sys.version_info < (3, 8):
    _is_python_greater_3_8 = False
else:
    _is_python_greater_3_8 = True


def is_peft_available() -> bool:
    return find_spec("peft") is not None


def is_unsloth_available() -> bool:
    return find_spec("unsloth") is not None


def is_accelerate_greater_20_0() -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version

        accelerate_version = version("accelerate")
    else:
        import pkg_resources

        accelerate_version = pkg_resources.get_distribution("accelerate").version
    return accelerate_version >= "0.20.0"


def is_transformers_greater_than(current_version: str) -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version

        _transformers_version = version("transformers")
    else:
        import pkg_resources

        _transformers_version = pkg_resources.get_distribution("transformers").version
    return _transformers_version > current_version


def is_torch_greater_2_0() -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version

        torch_version = version("torch")
    else:
        import pkg_resources

        torch_version = pkg_resources.get_distribution("torch").version
    return torch_version >= "2.0"


def is_wandb_available() -> bool:
    return find_spec("wandb") is not None


def is_xpu_available() -> bool:
    if is_accelerate_greater_20_0():
        import accelerate

        return accelerate.utils.is_xpu_available()
    else:
        if find_spec("intel_extension_for_pytorch") is None:
            return False
        try:
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except RuntimeError:
            return False


def is_npu_available() -> bool:
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if find_spec("torch") is None or find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    return hasattr(torch, "npu") and torch.npu.is_available()
