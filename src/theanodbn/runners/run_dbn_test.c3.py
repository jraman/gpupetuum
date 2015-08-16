'''
Bare bones run with small number of hidden units, pretrain epochs and finetune epochs.
Also, large batch_size.
'''

import logging
import os
import sys

path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path1)

from net import DBN

os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'

log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=1)

dataset_file = '../imnet_data/imnet_sample.n3.pkl'
label_file = '../imnet_data/label_sample.n3.txt'
pretrain_model_file = '../model/dbn.pretrain.test3.pkl'
finetuned_model_file = '../model/dbn.finetuned.test3.pkl'

hidden_layers_sizes = [128]
pretraining_epochs = 4
pretrain_lr = 0.05
cd_k = 1
finetune_training_epochs = 10
finetune_lr = 0.05
batch_size = 188 / 2 / 2
numpy_rng_seed = 4242

DBN.test_dbn(
    dataset_file=dataset_file,
    label_file=label_file,
    # pretrain_model_file=pretrain_model_file,
    pretrain_model_file='',
    finetuned_model_file=finetuned_model_file,
    hidden_layers_sizes=hidden_layers_sizes,
    pretraining_epochs=pretraining_epochs,
    pretrain_lr=pretrain_lr,
    k=cd_k,
    finetune_training_epochs=finetune_training_epochs,
    finetune_lr=finetune_lr,
    batch_size=batch_size,
    numpy_rng_seed=numpy_rng_seed,
)
