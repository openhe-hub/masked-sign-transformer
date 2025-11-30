"""Utilities to load pickle files saved with NumPy 2.x using older NumPy releases."""

import pickle
from typing import Any, BinaryIO


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps numpy._core references to numpy.core."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(file_obj: BinaryIO) -> Any:
    """Load a pickle file, falling back to NumPy 2.x compat mode when needed."""
    try:
        return pickle.load(file_obj)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        # rewind and retry with compat unpickler
        file_obj.seek(0)
        return _NumpyCompatUnpickler(file_obj).load()
