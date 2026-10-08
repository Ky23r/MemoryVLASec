from .base_memory_vla import BaseMemoryVLA

__all__ = ["BaseMemoryVLA", "SecureVLA"]


def __getattr__(name):
    if name == "SecureVLA":
        from .secure_vla import SecureVLA

        return SecureVLA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
