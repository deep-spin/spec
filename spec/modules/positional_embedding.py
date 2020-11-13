import math

from torch import nn

from spec.modules.positional_encoding import PositionalEncoding


class PositionalEmbedding(nn.Module):
    """Implements an Embedding Layer with absolute Positional Encoding.

    Args:
        vocab_size (int): vocabulary size
        size (int): embeddings size
        max_seq_len (int): hypotethical maximum sequence length
        dropout (float): dropout rate after applying PE (default: 0.)
        scale (bool): scale embeddings weights by sqrt(size) before PE
    """

    def __init__(
        self, vocab_size, size, max_seq_len=1000, dropout=0.0, scale=True
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, size)
        self.pe = PositionalEncoding(max_seq_len, size)
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x)
        if self.scale:
            x = math.sqrt(x.shape[-1]) * x
        x = self.pe(x)
        x = self.dropout(x)
        return x
