
class DbnConfig(object):
    dataset_file = '../../imnet_data/imnet_0.pkl'
    label_file = '../../imnet_data/0_label.txt'
    pretrain_model_file = '../../model/dbn.pretrain.c1000.pkl'
    finetuned_model_file = '../../model/dbn.finetuned.c1000.pkl'

    hidden_layers_sizes = [2048]
    pretraining_epochs = 100
    pretrain_lr = 0.05
    cd_k = 1
    finetune_training_epochs = 1000
    finetune_lr = 0.06
    num_mega_batches = 2
    batch_size = 203     # 65336 = 2 x 2 x 2 x 3 x 7 x 13 x 29 = 203 * 156 * 2
    numpy_rng_seed = 4242
    valid_size = batch_size * 70 * 3    # approx 30%
    test_size = batch_size * 70     # approx 10%
