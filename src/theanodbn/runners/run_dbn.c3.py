import logging
import os
import sys

path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path1)


def relpath(dirname):
    return os.path.abspath(os.path.join(path1, '..', dirname))

from net import DBN

os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'

log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.DEBUG)

dataset_file = relpath('imnet_data/imnet_sample.n3.pkl')
label_file = relpath('imnet_data/label_sample.n3.txt')
pretrain_model_file = relpath('model/dbn.pretrain.n3.pkl')
finetuned_model_file = relpath('model/dbn.finetuned.n3.pkl')

hidden_layers_sizes = [2048]
pretraining_epochs = 10
pretrain_lr = 0.05
cd_k = 1
finetune_training_epochs = 100
finetune_lr = 0.1
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
