import math

from torch import nn


def init_weights(dist_func, module, constant=None, dist='uniform', **kwargs):
    if dist not in dist_func.keys():
        raise Exception('distribution {} not found'.format(dist))
    for name, param in module.named_parameters():
        if param.dim() == 1:
            if constant is not None:
                nn.init.constant_(param, constant)
            else:
                stdv = 1.0 / math.sqrt(param.numel())
                nn.init.uniform_(param, -stdv, stdv)
        else:
            dist_func[dist](param, **kwargs)


def init_xavier(module, constant=None, dist='uniform', **kwargs):
    dist_func = {
        'uniform': nn.init.xavier_uniform_,
        'normal': nn.init.xavier_normal_
    }
    init_weights(dist_func, module, constant, dist, **kwargs)


def init_kaiming(module, constant=None, dist='uniform', **kwargs):
    dist_func = {
        'uniform': nn.init.kaiming_uniform_,
        'normal': nn.init.kaiming_normal_
    }
    init_weights(dist_func, module, constant, dist, **kwargs)
