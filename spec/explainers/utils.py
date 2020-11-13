import torch


def make_bow_matrix(word_ids, vocab_size, device=None):
    """
    Create a Bag of Words (BoW) matrix from word_ids.

    Args:
        word_ids (list of lists): a list of lists that contains the ids for each
            word in the sentence. Each id should be between [0, vocab_size).
            The number of lists should be equal to the batch_size, i.e.,
            len(word_ids) = batch_size
        vocab_size (int): the size of the words vocabulary
        device (int or torch.device): if not None, will sent the BoW matrix to
            this device. Default is None.

    Returns:
        torch.FloatTensor of shape (batch_size, vocab_size)
    """
    batch_size = len(word_ids)
    bow = torch.zeros(batch_size, vocab_size).to(device)
    for i, b_ids in enumerate(word_ids):
        for idx in b_ids:
            bow[i, idx] += 1.0
    return bow


def filter_word_ids_with_non_zero_probability(word_ids, probas, pad_id=None):
    """
    Filter out entries of word_ids that have exactly zero probability or are
    mapped to pad positions (if pad_id is not None).

    Args:
        word_ids (torch.LongTensor): tensor with word ids.
            Shape of (batch_size, seq_len)
        probas (torch.Tensor): tensor with probabilities for each word id. It
            can be also any other kind of measure of importance that could
            express 0 magnitude (e.g. norms). Shape of (batch_size, seq_len)
        pad_id (int): padding id in your vocabulary. If not None, word ids that
             are pad will be filtered out. Otherwise, all words are considered.
             Default is None
    Returns:
        a new list: valid word ids
    """
    valid_top_word_ids = []
    for seq_probas, seq_word_ids in zip(probas.tolist(), word_ids.tolist()):
        ids = []
        for prob, word_id in zip(seq_probas, seq_word_ids):
            if word_id != pad_id and prob > 0:
                ids.append(word_id)
        valid_top_word_ids.append(ids)
    return valid_top_word_ids
