'''
Bare bones run with small number of hidden units, pretrain epochs and finetune epochs.
When comparing with and without mega-batches, only the last mega-batch and its corresponding
numbers from the without scenario can be compared.

This config file can be used for either gpubatch (mini-batch with all data loaded in GPU memory),
or rambatch (mini-batch with all data loaded in CPU RAM and transferred back and forth to GPU memory)
'''


class DbnConfig(object):
    loglevel = 1

    dataset_file = '../../imnet_data/filelist.n3.txt'
    label_file = '../../imnet_data/label_select.n3.txt'
    pretrain_model_file = '../../model/dbn.pretrain.test3.pkl'
    finetuned_model_file = '../../model/dbn.finetuned.test3.pkl'
    load_from = 'disk'

    hidden_layers_sizes = [128]
    pretraining_epochs = 4
    pretrain_lr = 0.05
    cd_k = 1
    finetune_training_epochs = 10
    finetune_lr = 0.05
    num_mega_batches = 2
    batch_size = 188 / 2 / num_mega_batches
    numpy_rng_seed = 4242
    valid_size = 0      # unused
    test_size = 0       # unused
