from pathlib import Path

from spec import constants
from spec import opts
from spec.laymen.linear import LinearLayman
from spec.laymen.linear_entailment import LinearEntailmentLayman
from spec.laymen.linear_translation import TranslationLinearLayman

available_laymen = {
    'linear': LinearLayman,
    'linear_entailment': LinearEntailmentLayman,
    'linear_translation': TranslationLinearLayman
}


def build(options, fields_tuples, message_size, loss_weights):
    layman_cls = available_laymen[options.layman]
    layman = layman_cls(fields_tuples, message_size, options)
    layman.build_loss(loss_weights)
    if options.gpu_id is not None:
        layman = layman.cuda(options.gpu_id)
    return layman


def load_state(path, layman):
    layman_path = Path(path, constants.LAYMAN)
    layman.load(layman_path)


def load(path, fields_tuples, message_size, current_gpu_id):
    options = opts.load(path, name=constants.COMMUNICATION_CONFIG)

    # set gpu device to the current device
    options.gpu_id = current_gpu_id

    # hack: set dummy loss_weights (the correct values are going to be loaded)
    target_field = dict(fields_tuples)['target']
    loss_weights = None
    if options.loss_weights == 'balanced':
        loss_weights = [0] * (len(target_field.vocab) - 1)

    layman = build(options, fields_tuples, message_size, loss_weights)
    load_state(path, layman)
    return layman


def save(path, layman):
    layman_path = Path(path, constants.LAYMAN)
    layman.save(layman_path)
