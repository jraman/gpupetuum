
class DbnConfig(object):
    # paths are relative to location of this conf file
    dataset_file = '../imnet_data/imnet_sample.n3.pkl'
    label_file = '../imnet_data/label_sample.n3.txt'
    pretrain_model_file = '../model/dbn_batch.pretrain.test3.pkl'
    finetuned_model_file = '../model/dbn_batch.finetuned.test3.pkl'

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
