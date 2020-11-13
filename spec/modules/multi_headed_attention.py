# flake8: noqa: E501

import torch
from torch import nn

from spec.modules.attention import Attention


class MultiHeadedAttention(nn.Module):
    """Implements the key-value scaled self-attetion with multiple heads:
    https://arxiv.org/abs/1706.03762

    Input tensors are projected into new tensors by a linear transformation.
    We split each tensor by the number of heads. Then, attention is computed
    individually for each head.

    This implementation doesn't have an output projection as suggested by
    The Annotated Transformer blog post.

    Args:
        attn (Attention): base attention module used for each head.
        nb_heads (int): number of heads to split the projected inputs.
        query_size (int): last dimension of the query tensor.
        key_size (int): last dimension of the key tensor.
        value_size (int): last dimension of the value tensor.
        hidden_size (int): dimension of linear projection (must be divisible
            by `nb_heads`).
    """

    def __init__(
        self,
        attn,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size,
    ):
        super().__init__()

        # ensure hidden size is divisible by the nb of heads
        assert hidden_size % nb_heads == 0
        self.hidden_size = hidden_size
        self.nb_heads = nb_heads
        self.heads_size = hidden_size // nb_heads
        self.value_size = value_size

        self.proj_queries = nn.Linear(query_size, hidden_size)
        self.proj_keys = nn.Linear(key_size, hidden_size)
        self.proj_values = nn.Linear(value_size, hidden_size)

        self.attention = attn
        self.p_attn = None  # useful if you'd like to see attention weights

    def forward(self, queries, keys, values, mask=None):
        """Compute the attention between query, keys and values using multiple
        heads in parallel.

        Args:
            query (torch.Tensor): set of query vectors with shape of
                (batch_size, ..., target_len, hidden_size)
            keys (torch.Tensor): set of keys vectors with shape of
                (batch_size, ..., source_len, hidden_size)
            values (torch.Tensor, optional): set of values vectors with
                shape of: (batch_size, ..., source_len, hidden_size).
                If None, keys are treated as values. Default: None
            mask (torch.ByteTensor, optional): Tensor representing valid
                positions. If None, all positions are considered valid.
                Shape of (batch_size, target_len)

        Returns:
            torch.Tensor: combination of values and attention probabilities.
                Shape of (batch_size, ..., target_len, hidden_size)
            torch.Tensor: attention probabilities between query and keys.
                Shape of (batch_size, ..., target_len, source_len)
        """
        batch_size, _, _ = queries.shape

        # do all linear projections
        queries = self.proj_queries(queries)
        keys = self.proj_keys(keys)
        values = self.proj_values(values)

        # split heads
        queries = queries.view(batch_size, -1, self.nb_heads, self.heads_size).transpose(1, 2)  # NOQA
        keys = keys.view(batch_size, -1, self.nb_heads, self.heads_size).transpose(1, 2)  # NOQA
        values = values.view(batch_size, -1, self.nb_heads, self.heads_size).transpose(1, 2)  # NOQA

        # if a mask is provided, expand dims for the head axis
        # (see transpose above)
        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        # merge heads into batch dim
        queries = queries.reshape(batch_size * self.nb_heads, -1, self.heads_size)  # NOQA
        keys = keys.reshape(batch_size * self.nb_heads, -1, self.heads_size)
        values = values.reshape(batch_size * self.nb_heads, -1, self.heads_size)
        # tile mask
        if mask is not None:
            mask = mask.repeat_interleave(self.nb_heads, dim=0)

        # apply attention on all projected vectors
        x, self.p_attn = self.attention(queries, keys, values, mask=mask)

        # split head and batch dims
        x = x.view(batch_size, self.nb_heads, -1, self.heads_size)
        self.p_attn = self.p_attn.view(batch_size, -1, *self.p_attn.shape[-2:])

        # concat heads back
        o_attn = x.transpose(1, 2).reshape(batch_size, -1, self.nb_heads * self.heads_size)  # NOQA

        return o_attn, self.p_attn


if __name__ == "__main__":
    from spec.modules.utils import sequence_mask
    from spec.modules.scorer import (
        DotProductScorer,
        GeneralScorer,
        OperationScorer,
        MLPScorer,
    )

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    batch_size = 8
    source_len = 7
    target_len = 3
    query_size = 10
    key_size = 20
    value_size = key_size
    attn_size = 15
    nb_heads = 4
    hidden_size = nb_heads * 4
    source_mask = sequence_mask(torch.LongTensor([5, 3, 7, 4, 5, 4, 3, 6]))
    target_mask = sequence_mask(torch.LongTensor([2, 1, 2, 3, 3, 3, 2, 2]))

    # query vectors
    q = torch.randn(batch_size, query_size)

    # set of query vectors
    qs = torch.randn(batch_size, target_len, query_size)

    # keys vectors (a key vector for each encoder word)
    ks = torch.randn(batch_size, source_len, key_size)

    # keys vectors with same size as query vectors
    kq = torch.randn(batch_size, source_len, query_size)

    # values vectors (same shape as keys)
    vs = torch.randn(batch_size, source_len, key_size)

    # values vectors with same size as query vectors
    vq = torch.randn(batch_size, source_len, query_size)

    '''
    Self attentions:
    '''
    # multi headed self attention on target (decoder)
    scorer = DotProductScorer()
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        query_size,
        query_size,
        hidden_size
    )
    out, probs = attn(qs, qs, qs)
    assert list(out.shape) == [batch_size, qs.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, qs.shape[1], qs.shape[1]]

    # multi headed self attention on source (encoder)
    scorer = DotProductScorer()
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        query_size,
        query_size,
        hidden_size
    )
    out, probs = attn(kq, kq, kq)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    '''
    Masked multi headed self attentions:
    '''
    # masked multi headed self attention on target (decoder)
    scorer = DotProductScorer()
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        query_size,
        query_size,
        hidden_size
    )
    out, probs = attn(qs, qs, qs, mask=target_mask)
    assert list(out.shape) == [batch_size, qs.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, qs.shape[1], qs.shape[1]]

    # masked multi headed self attention on source (encoder)
    scorer = DotProductScorer()
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        query_size,
        query_size,
        hidden_size
    )
    out, probs = attn(kq, kq, kq, mask=source_mask)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    # masked multi headed self attention with different sentence lengths for
    # source and target
    scorer = DotProductScorer()
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        query_size,
        query_size,
        hidden_size
    )
    out, probs = attn(qs, kq, kq, mask=source_mask)
    assert list(out.shape) == [batch_size, qs.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, qs.shape[1], kq.shape[1]]

    '''
    Multi headed with different attentions
    '''
    # multi headed general attention on source (encoder)
    scorer = GeneralScorer(hidden_size // nb_heads, hidden_size // nb_heads)
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size
    )
    out, probs = attn(kq, ks, vs)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    # multi headed add attention on source (encoder)
    scorer = OperationScorer(
        hidden_size // nb_heads, hidden_size // nb_heads, attn_size, op='add'
    )
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size
    )
    out, probs = attn(kq, ks, vs)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    # multi headed concat attention on source (encoder)
    scorer = OperationScorer(
        hidden_size // nb_heads, hidden_size // nb_heads, attn_size, op='concat'
    )
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size
    )
    out, probs = attn(kq, ks, vs)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    '''
    Masked multi headed with different attentions
    '''
    # masked multi headed general attention
    scorer = GeneralScorer(hidden_size // nb_heads, hidden_size // nb_heads)
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size
    )
    out, probs = attn(kq, ks, vs, mask=source_mask)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    # masked multi headed concat attention
    scorer = OperationScorer(
        hidden_size // nb_heads, hidden_size // nb_heads, attn_size, op='add'
    )
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size
    )
    out, probs = attn(kq, ks, vs, mask=source_mask)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    # masked multi headed concat attention
    scorer = MLPScorer(
        hidden_size // nb_heads, hidden_size // nb_heads, layer_sizes=[5, 5]
    )
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        key_size,
        value_size,
        hidden_size
    )
    out, probs = attn(kq, ks, vs, mask=source_mask)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]

    # masked multi headed general self attention
    scorer = GeneralScorer(hidden_size // nb_heads, hidden_size // nb_heads)
    attn_base = Attention(scorer)
    attn = MultiHeadedAttention(
        attn_base,
        nb_heads,
        query_size,
        query_size,
        query_size,
        hidden_size
    )
    out, probs = attn(kq, kq, kq, mask=source_mask)
    assert list(out.shape) == [batch_size, kq.shape[1], hidden_size]
    assert list(probs.shape) == [batch_size, nb_heads, kq.shape[1], kq.shape[1]]
