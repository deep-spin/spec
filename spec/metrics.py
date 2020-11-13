import numpy as np


def total_variance_distance(gold_probas, pred_probas, reduction='mean'):
    """
    Calculate the total variance distance between two probability distributions.

    Args:
        gold_probas (2d array-like): gold probabilities.
            Shape of (batch_size, nb_labels)
        pred_probas (2d array-like): prediction probabilities.
            Shape of (batch_size, nb_labels)
        reduction (str): how to reduce the batch dimension.
            'mean': sum the tvd for each instance and divide it by batch_size
            'sum': sum the tvd for each instance
            None: return the tvd for each instance

    Returns:
        case reduction = 'sum', 'mean':
            float: tvd between gold_probas and pred_probas
        case reduction = None
            np.array of shape (batch_size)
    """
    assert len(gold_probas) == len(pred_probas)
    tvd = 0.5 * np.sum(np.abs(np.subtract(gold_probas, pred_probas)), axis=-1)
    if reduction == 'sum':
        return np.sum(tvd)
    elif reduction == 'mean':
        return np.mean(tvd)
    elif reduction is None:
        return tvd
    else:
        raise Exception('Reduction `{}` not available'.format(reduction))
