import json
from argparse import Namespace
from pathlib import Path

from spec.dataset import corpora
from spec import constants, explainers, laymen
from spec import models
from spec import optimizer
from spec import scheduler
from spec.dataset.vectors import available_embeddings
from spec.modules.attention import available_max_activations


def load(path, name=constants.CONFIG):
    config_path = Path(path, name)
    options = json.load(open(str(config_path), 'r'))
    return Namespace(**options)


def save(path, options, name=constants.CONFIG):
    config_path = Path(path, name)
    json.dump(vars(options), open(str(config_path), 'w'), indent=4)


def general_opts(parser):
    group = parser.add_argument_group('general')
    # Output
    group.add_argument('-o', '--output-dir',
                       type=str,
                       help='Will save files for this run under this dir. '
                            'If not specified, it will create a timestamp dir '
                            'inside `runs` dir.')

    # Data processing options
    group = parser.add_argument_group('random')
    group.add_argument('--seed',
                       type=int,
                       default=42,
                       help='Random seed')

    # Cuda
    group = parser.add_argument_group('gpu')
    group.add_argument('--gpu-id',
                       default=None,
                       type=int,
                       help='Use CUDA on the listed devices')
    # Logging
    group = parser.add_argument_group('logging')
    group.add_argument('--debug',
                       action='store_true',
                       help='Debug mode.')
    group.add_argument('--tensorboard',
                       action='store_true',
                       help='Whether to use tensorboardX for logging stats'
                            ' in addition to the regular output logger.')
    group.add_argument('--print-parameters-per-layer',
                       action='store_true',
                       help='Whether to print the number of parameters for'
                            'each layer.')

    # Save and load
    group = parser.add_argument_group('save-load')
    group.add_argument('--save',
                       type=str,
                       default=None,
                       help='Output dir for saving the model')
    group.add_argument('--load',
                       type=str,
                       default=None,
                       help='Input dir for loading the model/classifier')
    group.add_argument('--load-communication',
                       type=str,
                       default=None,
                       help='Input dir for loading the communication models')
    group.add_argument('--resume-epoch',
                       type=int,
                       default=None,
                       help='Resume training from a specific epoch saved in a '
                            'previous execution `runs/output-dir`')
    group.add_argument('--save-attention',
                       type=str,
                       default=None,
                       help='Output file for saving attended words.')


def preprocess_opts(parser):
    # Data options
    group = parser.add_argument_group('data')
    group.add_argument('--corpus',
                       type=str,
                       choices=list(corpora.available_corpora.keys()),
                       help='Which corpus reader to use.')
    group.add_argument('--train-path',
                       type=str,
                       help='Path to training directory with multiple example '
                            'files or path to a single file with multiple '
                            'examples separated by new lines')
    group.add_argument('--dev-path',
                       type=str,
                       help='Path to dev directory with multiple example '
                            'files or path to a single file with multiple '
                            'examples separated by new lines')
    group.add_argument('--test-path',
                       type=str,
                       help='Path to test directory with multiple example '
                            'files or path to a single file with multiple '
                            'examples separated by new lines')
    group.add_argument('--lazy-loading',
                       action='store_true',
                       help='Lazily load and iterate over the dataset.')

    # Truncation options
    group = parser.add_argument_group('data-pruning')
    group.add_argument('--max-length',
                       type=int,
                       default=10**12,  # max-length > 1 trillion? ow lord
                       help='Maximum sequence length')
    group.add_argument('--min-length',
                       type=int,
                       default=0,
                       help='Minimum sequence length.')

    # Dictionary options
    group = parser.add_argument_group('data-vocabulary')
    group.add_argument('--vocab-size',
                       type=int,
                       default=None,
                       help='Max size of the vocabulary.')
    group.add_argument('--vocab-min-frequency',
                       type=int,
                       default=1,
                       help='Min word frequency for vocabulary.')
    group.add_argument('--keep-rare-with-vectors',
                       action='store_true',
                       help='Keep words that occur less then min-frequency '
                            'but are in embeddings vocabulary.')
    group.add_argument('--add-embeddings-vocab',
                       action='store_true',
                       help='Add words from embeddings vocabulary to '
                            'the true vocabulary.')

    # Embeddings options
    group = parser.add_argument_group('data-embeddings')
    group.add_argument('--embeddings-format',
                       type=str,
                       default=None,
                       choices=list(available_embeddings.keys()),
                       help='Word embeddings format (source for MT). '
                            'See README for specific formatting instructions.')
    group.add_argument('--embeddings-format-target',
                       type=str,
                       default=None,
                       choices=list(available_embeddings.keys()),
                       help='Word embeddings format (target for MT). '
                            'See README for specific formatting instructions.')
    group.add_argument('--embeddings-path',
                       type=str,
                       help='Path to word embeddings file (source for MT).')
    group.add_argument('--embeddings-path-target',
                       type=str,
                       help='Path to word embeddings file (target for MT).')
    group.add_argument('--embeddings-binary',
                       action='store_true',
                       help='Whether the provided embeddings are in binary '
                            'format or not. By default they are treated as '
                            'not binary. (source for MT)')
    group.add_argument('--embeddings-binary-target',
                       action='store_true',
                       help='Whether the provided embeddings are in binary '
                            'format or not. By default they are treated as '
                            'not binary. (target for MT)')


def model_opts(parser):
    # Models options
    group = parser.add_argument_group('model-selection')
    group.add_argument('--model',
                       type=str,
                       default='simple_lstm',
                       choices=list(models.available_models.keys()),
                       help='Model architecture.')

    group = parser.add_argument_group('hyper-parameters')
    group.add_argument('--word-embeddings-size',
                       type=int,
                       default=100,
                       help='Size of word embeddings.')
    group.add_argument('--conv-size',
                       type=int,
                       default=100,
                       help='Size of convolution 1D. '
                            'a.k.a. number of channels.')
    group.add_argument('--kernel-size',
                       type=int,
                       default=7,
                       help='Size of the convolving kernel.')
    group.add_argument('--pool-length',
                       type=int,
                       default=3,
                       help='Size of pooling window.')
    group.add_argument('--cnn-dropout',
                       type=float,
                       default=0.0,
                       help='Dropout rate applied after conv+pool layers.')
    group.add_argument('--rnn-dropout',
                       type=float,
                       default=0.5,
                       help='Dropout rate applied after RNN layers.')
    group.add_argument('--emb-dropout',
                       type=float,
                       default=0.4,
                       help='Dropout rate applied after embedding layers.')
    group.add_argument('--rnn-type',
                       type=str,
                       default='rnn',
                       choices=['rnn', 'lstm', 'gru'],
                       help='RNN cell type: LSTM, GRU or a regular RNN.')
    group.add_argument('--bidirectional',
                       action='store_true',
                       help='Set RNNs to be bidirectional.')
    group.add_argument('--sum-bidir',
                       action='store_true',
                       help='Sum outputs of bidirectional states. '
                            'By default they are concatenated.')
    group.add_argument('--freeze-embeddings',
                       action='store_true',
                       help='Freeze embedding weights during training.')
    group.add_argument('--loss-weights',
                       type=str,
                       default='same',
                       choices=['same', 'balanced'],
                       help='Weights to penalize each class '
                            'in loss calculation. `same` will give each class '
                            'the same weights. `balanced` will give more '
                            'weight to minority classes.')
    group.add_argument('--hidden-size',
                       type=int,
                       nargs='+',
                       default=[100],
                       help='Number of neurons on the hidden layers. '
                            'If you pass more sizes, then more than one '
                            'hidden layer will be created. Please, take a '
                            'look at your selected model documentation '
                            'before setting this option.')
    group.add_argument('--attn-type',
                       type=str,
                       default='regular',
                       choices=['regular', 'multihead'],
                       help='The type of attention.')
    group.add_argument('--attn-scorer',
                       type=str,
                       default='dot_product',
                       choices=['dot_product', 'general', 'self_add', 'add',
                                'concat', 'mlp', 'lstm', 'conv'],
                       help='The way the attention similarity is calculated.')
    group.add_argument('--attn-hidden-size',
                       type=int,
                       default=100,
                       help='Hidden size for linear projections inside'
                            'the attention mechanism.')
    group.add_argument('--attn-dropout',
                       type=float,
                       default=0.0,
                       help='Dropout rate applied after attention.')
    group.add_argument('--attn-max-activation',
                       type=str,
                       choices=available_max_activations.keys(),
                       default='softmax',
                       help='Activation function used to map attention scores '
                            'to the simplex.')
    group.add_argument('--attn-nb-heads',
                       type=int,
                       default=4,
                       help='Number of heads for multiheaded attention.')
    group.add_argument('--attn-multihead-hidden-size',
                       type=int,
                       default=100,
                       help='Hidden size for linear projection for projecting'
                            'query, keys and values in multiheaded attention.')
    group.add_argument('--transformer-encoder-layers',
                       type=int,
                       default=1,
                       help='Number of stacked transformer encoder layers.')
    # continuous attention options:
    group.add_argument('--attn-cont-encoder',
                       type=str,
                       default='lstm',
                       choices=['lstm', 'average', 'last', 'conv'],
                       help='The encoder used in the continuous attention.')
    group.add_argument('--attn-cont-pool',
                       type=str,
                       default='last',
                       choices=['last', 'avg', 'avg_const', 'avg_minmax',
                                'other'],
                       help='The pooling used in the continuous encoder.')
    group.add_argument('--attn-nb-waves',
                       type=int,
                       default=16,
                       help='Number of sin/cos waves (or gaussians).')
    group.add_argument('--attn-freq',
                       type=int,
                       default=10000,
                       help='Frequency for sin/cos waves.')
    group.add_argument('--attn-power-basis',
                       action='store_true',
                       help='Whether to use power basis functions.')
    group.add_argument('--attn-wave-basis',
                       action='store_true',
                       help='Whether to use sin/cos basis functions.')
    group.add_argument('--attn-gaussian-basis',
                       action='store_true',
                       help='Whether to use gaussian basis functions.')
    group.add_argument('--attn-dynamic-nb-basis',
                       action='store_true',
                       help='Whether to use a dynamic nb of basis functions.')
    group.add_argument('--attn-consider-pad',
                       action='store_true',
                       help='Whether to consider pad positions in the '
                            'computation of the value function.')


def communication_opts(parser):
    group = parser.add_argument_group('explainer')
    group.add_argument('--explainer',
                       type=str,
                       default='encoded_attn',
                       choices=list(explainers.available_explainers.keys()),
                       help='Explainer model.')

    group = parser.add_argument_group('hyper-parameters')
    group.add_argument('--message-type',
                       type=str,
                       default='bow',
                       choices=['bow', 'embs', 'embs_sum', 'embs_mean',
                                'weighted_embs_sum', 'weighted_embs_mean'],
                       help='Message type returned from the explainer')
    group.add_argument('--explainer-attn-top-k',
                       type=int,
                       default=10,
                       help='How many higher topk entries to pick from the '
                            'attention weights.')
    group.add_argument('--explainer-random-type',
                       type=str,
                       default='uniform',
                       choices=['uniform', 'beta', 'shuffle', 'zero_max_out',
                                'first_states', 'last_states', 'mid_states'],
                       help='Method used to create random attention weights.')
    group.add_argument('--explainer-cheat-ratio',
                       type=float,
                       default=0.5,
                       help='Cheating ratio [0, 1] for the joint explainer.')
    group.add_argument('--explainer-lambda',
                       type=float,
                       default=2.0,
                       help='Lambda for the joint explainer loss')
    group.add_argument('--explainer-second-loss',
                       type=str,
                       default='none',
                       help='Second loss for joint explainer: ae, pred, lstm')
    group.add_argument('--explainer-idf',
                       type=str,
                       default='none',
                       help='Where to multiply idf: scores, embs or none.')
    group.add_argument('--explainer-ignore-top-words',
                       type=int,
                       default=0,
                       help='Ignore top x words.')

    group = parser.add_argument_group('layman')
    group.add_argument('--layman',
                       type=str,
                       default='linear',
                       choices=list(laymen.available_laymen.keys()),
                       help='Layman model.')

    group = parser.add_argument_group('communication')
    group.add_argument('--train-classifier',
                       action='store_true',
                       help='Whether to train the classifier together with the '
                            'layman during communication. By default, the '
                            'classifier is detached from the computation graph,'
                            ' i.e., it is treated as a leaf.')
    group.add_argument('--train-explainer',
                       action='store_true',
                       help='Whether to train the explainer together with the '
                            'layman during communication. By default, the '
                            'explainer is detached from the computation graph, '
                            'i.e., it is treated as a leaf.')
    group.add_argument('--freeze-classifier-params',
                       action='store_true',
                       help='Whether to freeze the classifier weights.')
    group.add_argument('--freeze-explainer-params',
                       action='store_true',
                       help='Whether to freeze the explainer weights.')
    group.add_argument('--use-gold-as-clf',
                       action='store_true',
                       help='Whether to use gold labels as the classifier'
                            'prediction (i.e. full acc classifier).')

    group = parser.add_argument_group('explanations')
    group.add_argument('--save-explanations',
                       type=str,
                       default=None,
                       help='Output file for saving explanations.')
    group.add_argument('--max-explanations',
                       type=int,
                       default=float('inf'),
                       help='Maximum number of explanations to be saved.')


def train_opts(parser):
    # Training loop options
    group = parser.add_argument_group('training')
    group.add_argument('--epochs',
                       type=int,
                       default=10,
                       help='Number of epochs for training.')
    group.add_argument('--shuffle',
                       action='store_true',
                       help='Shuffle train data before each epoch.')
    group.add_argument('--train-batch-size',
                       type=int,
                       default=64,
                       help='Maximum batch size for training.')
    group.add_argument('--dev-batch-size',
                       type=int,
                       default=64,
                       help='Maximum batch size for evaluating.')
    group.add_argument('--dev-checkpoint-epochs',
                       type=int,
                       default=1,
                       help='Perform an evaluation on dev set after X epochs.')
    group.add_argument('--save-checkpoint-epochs',
                       type=int,
                       default=1,
                       help='Save a checkpoint every X epochs. Set to 0 if '
                            'you dont want to save any checkpoint.')
    group.add_argument('--save-best-only',
                       action='store_true',
                       help='Save only when validation loss is improved. '
                       '(recommended)')
    group.add_argument('--early-stopping-patience',
                       type=int,
                       default=0,
                       help='Stop training if validation loss is not '
                            'improved after passing X epochs. By default '
                            'the early stopping procedure is not applied.')
    group.add_argument('--restore-best-model',
                       action='store_true',
                       help='Whether to restore the model state from '
                            'the epoch with the best validation loss found. '
                            'If False, the model state obtained at the last '
                            'step of training is used.')
    group.add_argument('--final-report', action='store_true',
                       help='Whether to report a table with the stats history '
                            'for train/dev/test set after training.')

    # Optimization options
    group = parser.add_argument_group('training-optimization')
    group.add_argument('--optimizer',
                       default='sgd',
                       choices=list(optimizer.available_optimizers.keys()),
                       help='Optimization method.')
    group.add_argument('--learning-rate',
                       type=float,
                       default=None,
                       help='Starting learning rate. '
                            'Let unseted to use default values.')
    group.add_argument('--weight-decay',
                       type=float,
                       default=None,
                       help='L2 penalty. Used for all algorithms. '
                            'Let unseted to use default values.')
    group.add_argument('--lr-decay',
                       type=float,
                       default=None,
                       help='Learning reate decay. Used only for: '
                            'adagrad. '
                            'Let unseted to use default values.')
    group.add_argument('--rho',
                       type=float,
                       default=None,
                       help='Coefficient used for computing a running '
                            'average of squared. Used only for: '
                            'adadelta. '
                            'Let unseted to use default values.')
    group.add_argument('--beta0',
                       type=float,
                       default=None,
                       help='Coefficient used for computing a running '
                            'averages of gradient and its squared. '
                            'Used only for: adam, sparseadam, adamax. '
                            'Let unseted to use default values.')
    group.add_argument('--beta1',
                       type=float,
                       default=None,
                       help='Coefficient used for computing a running '
                            'averages of gradient and its squared. '
                            'Used only for: adam, sparseadam, adamax. '
                            'Let unseted to use default values.')
    group.add_argument('--momentum',
                       type=float,
                       default=None,
                       help='Momentum factor. Used only for: '
                            'sgd and rmsprop. '
                            'Let unseted to use default values.')
    group.add_argument('--nesterov',
                       action='store_true',
                       help='Enables Nesterov momentum. Used only for sgd.')
    group.add_argument('--alpha',
                       type=float,
                       default=None,
                       help='Smoothing constant. Used only for: rmsprop. '
                            'Let unseted to use default values.')
    group.add_argument('--amsgrad',
                       action='store_true',
                       help='Whether to use the AMSGrad variant for '
                            'Adam and AdamW.')
    # optimizer with learning rate decay during training steps
    group.add_argument('--lr-step-decay',
                       default=None,
                       choices=list(optimizer.available_step_decays.keys()),
                       help='Method used for learning rate decay during '
                            'training iterations.')
    group.add_argument('--warmup-steps',
                       type=int,
                       default=None,
                       help='LR will increase until this number of steps.')
    group.add_argument('--decay-steps',
                       type=int,
                       default=None,
                       help='Scale LR every `decay-steps` steps')

    # LRScheduler options
    group = parser.add_argument_group('training-learning-rate-scheduler')
    group.add_argument('--scheduler',
                       default=None,
                       choices=list(scheduler.available_schedulers.keys()),
                       help='Learning rate scheduler method.')
    group.add_argument('--step-size',
                       type=int,
                       default=None,
                       help='Period of learning rate decay.')
    group.add_argument('--gamma',
                       type=float,
                       default=0.1,
                       help='Multiplicative factor of learning rate decay.')
    group.add_argument('--eta-min',
                       type=float,
                       default=1e-7,
                       help='Minimum learning rate.')
    group.add_argument('--t-max',
                       type=float,
                       default=None,
                       help='Maximum number of iterations. If None and cosine '
                            'annealing is selected, it will be set to the '
                            'size of the dataset.')


def predict_opts(parser):
    # Prediction options
    group = parser.add_argument_group('predicting')
    group.add_argument('--text', type=str, default=None,
                       help='A text to be predicted. '
                            'The text will be splited into sentences '
                            'ending with .?!')
    group.add_argument('--prediction-type',
                       type=str,
                       default='classes',
                       choices=['classes', 'probas'],
                       help='Whether to predict classes or probabilities.')


def get_default_args():
    """Useful for creating a Classifier object with default params."""
    import argparse
    parser = argparse.ArgumentParser()
    general_opts(parser)
    preprocess_opts(parser)
    model_opts(parser)
    communication_opts(parser)
    train_opts(parser)
    predict_opts(parser)
    args = parser.parse_args()
    return vars(args)
