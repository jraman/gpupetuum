'''
Bare bones run with small number of hidden units, pretrain epochs and finetune epochs.
When comparing with and without mega-batches, only the last mega-batch and its corresponding
numbers from the without scenario can be compared.
'''

import importlib
import logging
import os
import sys

# THEANO_FLAGS must be set prior to importing theano
if not os.environ.get('THEANO_FLAGS', None):
    os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'


# add path to dbn_batch
path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path1 not in sys.path:
    sys.path.append(path1)
from net import dbn_batch


def relpath(basepath, datafilepath):
    'helper function to resolve datafilepath'
    return os.path.abspath(os.path.join(path1, datafilepath))

conf_filepath = sys.argv[1]
conf_dir, conf_basename = os.path.dirname(conf_filepath), os.path.basename(conf_filepath)
if conf_dir not in sys.path:
    sys.path.append(conf_dir)

conf_module = importlib.import_module(conf_basename.replace('.py', ''))
conf = conf_module.DbnConfig

log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=1)


runner = dbn_batch.DbnMegaBatch(
    dataset_file=relpath(conf_dir, conf.dataset_file),
    label_file=relpath(conf_dir, conf.label_file),
    pretrain_model_file=relpath(conf_dir, conf.pretrain_model_file),
    finetuned_model_file=relpath(conf_dir, conf.finetuned_model_file),
    hidden_layers_sizes=conf.hidden_layers_sizes,
    pretraining_epochs=conf.pretraining_epochs,
    pretrain_lr=conf.pretrain_lr,
    k=conf.cd_k,
    finetune_training_epochs=conf.finetune_training_epochs,
    finetune_lr=conf.finetune_lr,
    num_mega_batches=conf.num_mega_batches,
    batch_size=conf.batch_size,
    numpy_rng_seed=conf.numpy_rng_seed,
    valid_size=conf.valid_size,
    test_size=conf.test_size,
)

runner.run()
