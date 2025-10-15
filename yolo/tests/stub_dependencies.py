"""
Helpers to provide lightweight stubs for optional heavy dependencies during testing.
"""

from __future__ import annotations

import types
import sys


def ensure_module(name: str) -> types.ModuleType:
    """Create a dummy module if it does not already exist."""
    if name in sys.modules:
        return sys.modules[name]

    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def install_torch_stub() -> None:
    """Install a minimal torch stub sufficient for unit tests."""
    if "torch" in sys.modules:
        return

    torch_stub = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data=None):
            self.data = data

    def equal(a, b):
        return a == b

    def from_numpy(arr):
        return arr

    torch_stub.Tensor = _Tensor
    torch_stub.equal = equal
    torch_stub.from_numpy = from_numpy

    nn_module = types.ModuleType("torch.nn")
    functional_module = types.ModuleType("torch.nn.functional")
    nn_module.functional = functional_module
    torch_stub.nn = nn_module

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_module
    sys.modules["torch.nn.functional"] = functional_module


def install_optional_stubs() -> None:
    """Install stubs for optional third-party libraries used by legacy modules."""
    modules = (
        "cv2",
        "torchvision",
        "torchvision.ops",
        "torchvision.transforms",
        "ultralytics",
        "tesserocr",
        "scipy",
        "scipy.optimize",
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.patches",
    )

    for name in modules:
        module = ensure_module(name)
        parent_name = name.rpartition(".")[0]
        if parent_name:
            parent = ensure_module(parent_name)
            setattr(parent, name.split(".")[-1], module)

        if name == "scipy.optimize":
            def _linear_sum_assignment(cost_matrix, *args, **kwargs):  # pragma: no cover - stub
                n_rows = len(cost_matrix)
                rows = list(range(n_rows))
                cols = list(range(len(cost_matrix[0]) if n_rows else 0))
                return rows, cols[:n_rows]

            module.linear_sum_assignment = _linear_sum_assignment

        if name == "ultralytics":
            class _YOLO:  # pragma: no cover - stub
                def __init__(self, *args, **kwargs):
                    pass

                def predict(self, *args, **kwargs):
                    return []

            module.YOLO = _YOLO

        if name == "torchvision.transforms":
            class _Compose(list):  # pragma: no cover - stub
                def __call__(self, value):
                    result = value
                    for func in self:
                        result = func(result)
                    return result

            module.Compose = _Compose

        if name == "matplotlib.pyplot":
            class _Figure:  # pragma: no cover - stub
                def __init__(self):
                    pass

            class _Axes:
                def __init__(self):
                    pass

                def imshow(self, *args, **kwargs):
                    pass

                def set_title(self, *args, **kwargs):
                    pass

                def axis(self, *args, **kwargs):
                    pass

                def add_patch(self, *args, **kwargs):
                    pass

                def plot(self, *args, **kwargs):
                    pass

                def text(self, *args, **kwargs):
                    pass

            def _subplots(*args, **kwargs):
                return _Figure(), _Axes()

            module.subplots = _subplots
            module.show = lambda *args, **kwargs: None
            module.tight_layout = lambda *args, **kwargs: None

        if name == "matplotlib.patches":
            class _Rectangle:  # pragma: no cover - stub
                def __init__(self, *args, **kwargs):
                    pass

            module.Rectangle = _Rectangle

        if name == "tesserocr":
            class _PyTessBaseAPI:  # pragma: no cover - stub
                def __init__(self, *args, **kwargs):
                    pass

                def End(self):
                    pass

                def SetImage(self, *args, **kwargs):
                    pass

                def GetUTF8Text(self):
                    return ""

                def SetPageSegMode(self, *args, **kwargs):
                    pass

                def SetVariable(self, *args, **kwargs):
                    pass

            module.PyTessBaseAPI = _PyTessBaseAPI
            module.PSM = types.SimpleNamespace(
                SINGLE_BLOCK=0,
                SINGLE_CHAR=1,
                SINGLE_WORD=2,
                SINGLE_LINE=3,
            )


def install_test_stubs() -> None:
    """Install all dependency stubs required for lightweight unit tests."""
    install_torch_stub()
    install_optional_stubs()
