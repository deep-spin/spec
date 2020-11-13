import random

import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

bs, ts = 2, 8
vocab_size = 10

wids = [[1, 2, 8, 4, 5, 1, 9, 4], [1, 2, 6, 2, 1, 2, 9, 0]]
wids = torch.LongTensor(wids)

probax = torch.rand(bs, ts, requires_grad=True)
probas = probax ** 2

mask = torch.ne(wids, 0).int()
bids = torch.arange(bs).unsqueeze(-1).expand(-1, ts).flatten()

idxs = torch.stack((bids, wids.flatten()), dim=0)
vals = mask.float() * probas
vals = vals.flatten()
size = torch.Size([bs, vocab_size])
bow = torch.sparse.FloatTensor(idxs, vals, size).to_dense()

idxs2 = torch.nn.functional.one_hot(wids)
vals2 = idxs2 * probas.unsqueeze(-1) * mask.unsqueeze(-1)
bow2 = torch.sum(vals2, dim=1)
assert torch.allclose(bow, bow2.float())

emb = torch.nn.Embedding(vocab_size, vocab_size, padding_idx=0, sparse=True)
emb.weight.requires_grad_(False)
emb.weight.data = torch.eye(vocab_size)
emb.weight[0].fill_(0)
vals3 = emb(wids) * probas.unsqueeze(-1)
bow3 = vals3.sum(1)
assert torch.allclose(bow, bow3.float())


bow.sum().backward(retain_graph=True)
grad_1 = probax.grad.clone()

probax.grad.zero_()
bow2.sum().backward(retain_graph=True)
grad_2 = probax.grad.clone()

probax.grad.zero_()
bow3.sum().backward(retain_graph=True)
grad_3 = probax.grad.clone()

import ipdb; ipdb.set_trace()
assert torch.allclose(grad_1, grad_2)
assert torch.allclose(grad_1, grad_3)
