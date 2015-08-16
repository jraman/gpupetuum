
import logging
import os
import sys

path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path1)


def relpath(dirname):
    return os.path.abspath(os.path.join(path1, '..', dirname))

from net import dbn_batch

os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'

log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.DEBUG)

dataset_file = relpath('imnet_data/imnet_sample.n3.pkl')
label_file = relpath('imnet_data/label_sample.n3.txt')
pretrain_model_file = relpath('model/dbn_batch.pretrain.test3.pkl')
finetuned_model_file = relpath('model/dbn_batch.finetuned.test3.pkl')

hidden_layers_sizes = [2048]
pretraining_epochs = 100
pretrain_lr = 0.05
cd_k = 1
finetune_training_epochs = 100
finetune_lr = 0.01
num_mega_batches = 2
batch_size = 188 / 2 / num_mega_batches
numpy_rng_seed = 4242
valid_size = batch_size
test_size = batch_size

runner = dbn_batch.DbnMegaBatch(
    dataset_file=dataset_file,
    label_file=label_file,
    pretrain_model_file=pretrain_model_file,
    finetuned_model_file=finetuned_model_file,
    hidden_layers_sizes=hidden_layers_sizes,
    pretraining_epochs=pretraining_epochs,
    pretrain_lr=pretrain_lr,
    k=cd_k,
    finetune_training_epochs=finetune_training_epochs,
    finetune_lr=finetune_lr,
    num_mega_batches=num_mega_batches,
    batch_size=batch_size,
    numpy_rng_seed=numpy_rng_seed,
    valid_size=valid_size,
    test_size=test_size,
)

runner.run()
