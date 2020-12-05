from .guard import ShapeGuard

try:
    import torch  # type: ignore
    torch.Tensor.sg = ShapeGuard.singleton_guard
except ImportError:
    pass
