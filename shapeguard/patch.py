from .guard import sg

try:
    import torch

    # But this will give type errors since Tensor doesn't have an `sg`
    torch.Tensor.sg = sg
except ImportError:
    pass
