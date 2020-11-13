import numpy as np
import torch


def unmask(tensor, mask, cut_length=0):
    """
    Unmask a tensor and convert it back to a list of lists.
    Args:
        tensor (torch.Tensor): tensor with shape (bs, max_len, ...)
        mask (torch.Tensor): tensor with shape (bs, max_len) where 1 (or True)
            indicates a valid position, and 0 (or False) otherwise
        cut_length (int): remove the last `cut_length` elements from the tensor.
            In practice, the lengths calculated from the mask are going to be
            subtracted by `cut_length`. This is useful when you have <bos> and
            <eos> tokens in your words field and the mask was computed with
            words != <pad>. Default is 0, i.e., no cut
    Returns:
         a list of lists with variable length
    """
    lengths = mask.int().sum(dim=-1)
    if cut_length > 0:
        lengths -= cut_length
    lengths = lengths.tolist()
    return [x[:lengths[i]].tolist() for i, x in enumerate(tensor)]


def unroll(list_of_lists, rec=False):
    """
    Unroll a list of lists
    Args:
        list_of_lists (list): a list that contains lists
        rec (bool): unroll recursively
    Returns:
        a single list
    """
    if not isinstance(list_of_lists[0], (np.ndarray, list)):
        return list_of_lists
    new_list = [item for l in list_of_lists for item in l]
    if rec and isinstance(new_list[0], (np.ndarray, list)):
        return unroll(new_list, rec=rec)
    return new_list


def freeze_all_module_params(module):
    """
    Set requires_grad of all params of a given module to False
    Args:
        module (torch.nn.Moduke): a torch.nn's module
    """
    for param in module.parameters():
        param.required_grad = False
