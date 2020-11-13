from torchtext.data import BucketIterator
import torch

from spec.dataset.modules.iterator import LazyBucketIterator


def build(dataset, device, batch_size, is_train, lazy=False):
    device = None if device is None else torch.device(device)
    iterator_cls = LazyBucketIterator if lazy else BucketIterator
    iterator = iterator_cls(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,
        sort_key=dataset.sort_key,
        sort=False,
        sort_within_batch=is_train,
        # shuffle batches
        shuffle=is_train,
        device=device,
        train=is_train
    )
    return iterator
