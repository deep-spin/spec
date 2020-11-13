import copy
import torch


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions.
    Args:
        size (int): squared tensor size
    Returns:
        torch.Tensor: (size, size)
    """
    return torch.tril(torch.ones(size, size, dtype=torch.uint8))


def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths.
    Args:
        lengths (torch.Tensor): lengths with shape (bs,)
        max_len (int, optional): max sequence length.
            if None it will be setted to lengths.max()
    Returns:
        torch.Tensor: (bs, max_len)
    """
    if max_len is None:
        max_len = lengths.max()
    aranges = torch.arange(max_len).repeat(lengths.shape[0], 1)
    aranges = aranges.to(lengths.device)
    return aranges < lengths.unsqueeze(1)


def neighbours_mask(size, window_size):
    """Mask for neighbour positions.
    Args:
        size (int): squared tensor size
        window_size (int): how many elements to be considered as valid around
            the ith element (including ith).
    Returns:
        torch.Tensor: (size, size)
    """
    z = torch.ones(size, size, dtype=torch.uint8)
    mask = (torch.triu(z, diagonal=1 + window_size // 2)
            + torch.tril(z, diagonal=- window_size // 2))
    return z - mask


def unsqueeze_as(tensor, as_tensor, dim=-1):
    """Expand new dimensions based on a template tensor along `dim` axis.
    Args:
        Args:
        tensor (torch.Tensor): tensor with shape (bs, ..., d1)
        as_tensor (torch.Tensor): tensor with shape (bs, ..., n, ..., d2)
    Returns:
        torch.Tensor: (bs, ..., 1, ..., d1)
    """
    x = tensor
    while x.dim() < as_tensor.dim():
        x = x.unsqueeze(dim)
    return x


def make_mergeable_tensors(t1, t2):
    """Expand a new dimension in t1 and t2 and expand them so that both
    tensors will have the same number of timesteps.
    Args:
        t1 (torch.Tensor): tensor with shape (bs, ..., m, d1)
        t2 (torch.Tensor): tensor with shape (bs, ..., n, d2)
    Returns:
        torch.Tensor: (bs, ..., m, n, d1)
        torch.Tensor: (bs, ..., m, n, d2)
    """
    assert t1.dim() == t2.dim()
    assert t1.dim() >= 3
    assert t1.shape[:-2] == t2.shape[:-2]
    # new_shape = [-1, ..., m, n, -1]
    new_shape = [-1 for _ in range(t1.dim() + 1)]
    new_shape[-3] = t1.shape[-2]  # m
    new_shape[-2] = t2.shape[-2]  # n
    # (bs, ..., m, d1) -> (bs, ..., m, 1, d1) -> (bs, ..., m, n, d1)
    new_t1 = t1.unsqueeze(-2).expand(new_shape)
    # (bs, ..., n, d2) -> (bs, ..., 1, n, d2) -> (bs, ..., m, n, d2)
    new_t2 = t2.unsqueeze(-3).expand(new_shape)
    return new_t1, new_t2
