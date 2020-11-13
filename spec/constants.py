# lowercased special tokens
UNK = '<unk>'
PAD = '<pad>'
START = '<bos>'
STOP = '<eos>'

# special tokens id (don't edit this order)
UNK_ID = 0
PAD_ID = 1

# this should be set later after building fields
TARGET_PAD_ID = -1

# output_dir
OUTPUT_DIR = 'runs'

# default filenames
CONFIG = 'config.json'
COMMUNICATION_CONFIG = 'communication_config.json'
DATASET = 'dataset.torch'
MODEL = 'model.torch'
EXPLAINER = 'explainer.torch'
LAYMAN = 'layman.torch'
OPTIMIZER = 'optim.torch'
EXPLAINER_OPTIMIZER = 'optim_explainer.torch'
LAYMAN_OPTIMIZER = 'optim_layman.torch'
SCHEDULER = 'scheduler.torch'
TRAINER = 'trainer.torch'
VOCAB = 'vocab.torch'
PREDICTIONS = 'predictions.txt'
