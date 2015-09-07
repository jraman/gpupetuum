'''
When comparing with and without mega-batches, only the last mega-batch and its corresponding
numbers from the without scenario can be compared.
'''

import argparse
import functools
import importlib
import logging
import os
import sys

# THEANO_FLAGS must be set prior to importing theano
if not os.environ.get('THEANO_FLAGS', None):
    os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'


# add path to dbn_batch (assumed to be in ..)
path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path1 not in sys.path:
    sys.path.append(path1)
from net import dbn_batch


def _myargparse():
    help_formatter = functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=104)
    parser = argparse.ArgumentParser(
        description='DBN Runner',
        formatter_class=help_formatter)
    parser.add_argument('-c', '--conf-filepath', action='store', help='config file with path', required=True)
    parser.add_argument('--finetune-lr', action='store', type=float, help='finetune learning rate')
    args = parser.parse_args()
    return args


def relpath(basepath, datafilepath):
    '''Helper function to resolve datafilepath.
     * If datafilepath starts with / (i.e. is an absolutepath), then return datafilepath unchanged.
     * If datafilepath is None (e.g. specify None for model_file to skip saving it), then return None.
     * Else, relative path is relative to location of the dbn_batch file.
    '''
    return datafilepath and os.path.abspath(os.path.join(path1, datafilepath))

args = _myargparse()
conf_dir, conf_basename = os.path.dirname(args.conf_filepath), os.path.basename(args.conf_filepath)
if conf_dir not in sys.path:
    sys.path.append(conf_dir)

conf_module = importlib.import_module(conf_basename.replace('.py', ''))
conf = conf_module.DbnConfig

log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=conf.loglevel)


finetune_lr = args.finetune_lr or conf.finetune_lr
runner = dbn_batch.DbnMegaBatch(
    dataset_file=relpath(conf_dir, conf.dataset_file),
    label_file=relpath(conf_dir, conf.label_file),
    pretrain_model_file=relpath(conf_dir, conf.pretrain_model_file),
    finetuned_model_file=relpath(conf_dir, conf.finetuned_model_file),
    load_from=conf.load_from,
    hidden_layers_sizes=conf.hidden_layers_sizes,
    pretraining_epochs=conf.pretraining_epochs,
    pretrain_lr=conf.pretrain_lr,
    k=conf.cd_k,
    finetune_training_epochs=conf.finetune_training_epochs,
    finetune_lr=finetune_lr,
    num_mega_batches=conf.num_mega_batches,
    batch_size=conf.batch_size,
    numpy_rng_seed=conf.numpy_rng_seed,
    valid_size=conf.valid_size,
    test_size=conf.test_size,
)

runner.run()
