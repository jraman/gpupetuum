'''
Dataset: TIMIT
Number of layers: 6
Number of units per layer: 2048
'''
import logging


class DbnConfig(object):
    loglevel = logging.INFO

    dataset_file = '../../timit_data/Train.fea.pkl'
    label_file = '../../timit_data/Train.label'
    # pretrain_model_file = '../../model/dbn.pretrain.c2001.pkl'
    # finetuned_model_file = '../../model/dbn.finetuned.c2001.pkl'
    pretrain_model_file = None
    finetuned_model_file = None

    load_from = 'RAM'

    num_layers = 6
    hidden_layers_sizes = [2048] * num_layers
    pretraining_epochs = 0
    pretrain_lr = 0.1
    cd_k = 1
    finetune_training_epochs = 100
    finetune_lr = 0.3
    num_mega_batches = 1
    batch_size = 20639     # 1031950 = 2 * 5 * 5 * 20639
    numpy_rng_seed = 4242
    valid_size = 0
    test_size = 0
    continue_run = False
    start_model_file = None
    pretrain_epoch_start = 0
    finetune_epoch_start = 0
