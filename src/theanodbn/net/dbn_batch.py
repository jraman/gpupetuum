'''
Run DBN by loading minibatches into GPU memory.
Needed when all of the data does not fit into GPU memory.

Notes:
 * validation and test sets overlap with training set.
 * We use parts of the last megabatch as validation and test sets.
'''

import logging
import numpy as np
import os
import pydot
import theano
import theano.tensor as T
import timeit

import DBN
import nn_util


assert pydot


# Add another level to the logger
DEBUG2_LEVEL_NUM = 9
logging.addLevelName(DEBUG2_LEVEL_NUM, "DEBUG2")


def debug2(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(DEBUG2_LEVEL_NUM):
        self._log(DEBUG2_LEVEL_NUM, message, args, **kws)
logging.Logger.debug2 = debug2
logging.debug2 = logging.root.debug2


class FeatureSet(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.yf = None


class DbnMegaBatch(object):
    def __init__(
        self,
        dataset_file,
        label_file,
        pretrain_model_file,
        finetuned_model_file,
        hidden_layers_sizes=[1024],
        pretraining_epochs=100,
        pretrain_lr=0.01,
        k=1,
        finetune_training_epochs=1000,
        finetune_lr=0.1,
        num_mega_batches=1,
        batch_size=10,
        numpy_rng_seed=4242,
        valid_size=None,
        test_size=None,
    ):
        """
        Demonstrates how to train and test a Deep Belief Network.

        :type finetune_lr: float
        :param finetune_lr: learning rate used in the finetune stage
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining
        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        :type k: int
        :param k: number of Gibbs steps in CD/PCD
        :type training_epochs: int
        :param training_epochs: maximal number of iterations to run the optimizer
        :type dataset_file: string
        :param dataset_file: path to the pickled dataset file
        :type batch_size: int
        :param batch_size: the size of a minibatch
        """
        logging.info('hidden_layers_sizes={}'.format(hidden_layers_sizes))
        logging.info('pretraining_epochs={}'.format(pretraining_epochs))
        logging.info('pretrain_lr={}'.format(pretrain_lr))
        logging.info('CD-k={}'.format(k))
        logging.info('finetune_training_epochs={}'.format(finetune_training_epochs))
        logging.info('finetune_lr={}'.format(finetune_lr))
        logging.info('num_mega_batches={}'.format(num_mega_batches))
        logging.info('batch_size={}'.format(batch_size))
        logging.info('numpy_rng seed={}'.format(numpy_rng_seed))
        logging.info('valid_size={}'.format(valid_size))
        logging.info('test_size={}'.format(test_size))

        self.dataset_file = dataset_file
        self.label_file = label_file
        self.pretrain_model_file = pretrain_model_file
        self.finetuned_model_file = finetuned_model_file
        self.hidden_layers_sizes = hidden_layers_sizes
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.k = k
        self.finetune_training_epochs = finetune_training_epochs
        self.finetune_lr = finetune_lr
        self.num_mega_batches = num_mega_batches
        self.batch_size = batch_size
        self.numpy_rng_seed = numpy_rng_seed
        self.valid_size = valid_size
        self.test_size = test_size

        self.num_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.num_minibatches = 0                # across all mega
        self.num_minibatches_in_mega = 0        # num batches in a mega batch
        self.mm_train_set = FeatureSet()
        self.dbn = None

    def run(self):
        logging.info('THEANO_FLAGS={}'.format(os.getenv('THEANO_FLAGS')))
        self.load_data_mm()
        self.build_model()
        self.load_data_th()
        self.pretrain()
        self.finetune()

    def load_data_mm(self):
        '''Load all of input data into main memory.
        Set up theano shared variables so that data can be moved from main memory to GPU memory in mega batches.
        '''
        with open(self.dataset_file, 'rb') as fobj:
            self.mm_train_set.x = np.array([x for x in nn_util.load_pickle_file(fobj)], dtype=theano.config.floatX)

        with open(self.label_file, 'r') as ff:
            self.mm_train_set.y = np.array([int(line.strip()) for line in ff], dtype=theano.config.floatX)

        # theano requires that the labels be in the range [0, L), where L is the number of unique labels.
        unique_labels = set(self.mm_train_set.y)
        if sorted(unique_labels) == list(unique_labels):
            logging.info('keeping the labels as is')
            label2idx = dict((ii, ii) for ii in unique_labels)
        else:
            logging.info('translating labels to range(N)')
            label2idx = dict((ll, ii) for ii, ll in enumerate(unique_labels))
        self.mm_train_set.y = np.array([label2idx[ll] for ll in self.mm_train_set.y], dtype=theano.config.floatX)

        self.num_samples = self.mm_train_set.x.shape[0]
        self.num_features = self.mm_train_set.x.shape[1]
        self.num_classes = len(set(self.mm_train_set.y))
        logging.info('num_samples={}, num_features={}, num_classes={}'.format(
            self.num_samples, self.num_features, self.num_classes))

        assert self.num_samples == self.mm_train_set.y.shape[0], "Num rows in X and Y don't match"

        assert self.num_samples % self.batch_size == 0, 'num_samples not a int multiple of batch_size'
        self.num_minibatches = self.num_samples / self.batch_size
        self.num_minibatches_in_mega = self.num_samples / self.batch_size / self.num_mega_batches
        logging.info('num_minibatches={}, num_minibatches_in_mega={}'.format(
            self.num_minibatches, self.num_minibatches_in_mega))

        msg = 'num_samples must be an integer multiple of num_mega_batches'
        assert self.num_samples / self.num_mega_batches * self.num_mega_batches == self.num_samples, msg

    def load_data_th(self, borrow=True):
        self.th_train_set = FeatureSet()
        mega_batch_y = np.zeros(self.num_samples / self.num_mega_batches)
        self.th_train_set.yf = theano.shared(
            np.asarray(mega_batch_y, dtype=theano.config.floatX),
            name='train_set.y',
            borrow=borrow,
        )
        self.th_train_set.y = T.cast(self.th_train_set.yf, 'int32')

        # set up theano shared variables
        mega_batch_x = np.zeros((self.num_samples / self.num_mega_batches, self.num_features))
        self.th_train_set.x = theano.shared(
            np.asarray(mega_batch_x, dtype=theano.config.floatX),
            name='train_set.X',
            borrow=borrow,
        )
        logging.info('set up th_train_set.x with shape: {}'.format(
            self.th_train_set.x.get_value(borrow=borrow).shape))

        logging.debug2('self.th_train_set.y[:20]={}'.format(self.th_train_set.y.eval()[:20]))
        # with replacement
        # np.random.seed(4242)
        # self.valid_size = self.valid_size or round(0.3 * self.num_samples)
        # valid_set_idx = np.random.choice(self.num_samples, self.valid_size)
        # self.valid_set.x, self.valid_set.y = self.train_set.x[valid_set_idx], self.train_set.y[valid_set_idx]
        # logging.info('Selected {} rows as validation set'.format(self.valid_set.x.shape[0]))

        # np.random.seed(4343)
        # self.test_size = self.test_size or (0.2 * self.num_samples)
        # test_set_idx = np.random.choice(self.num_samples, self.test_size)
        # self.test_set.x, self.test_set.y = self.train_set.x[test_set_idx], self.train_set.y[test_set_idx]
        # logging.info('Selected {} rows as test set'.format(self.test_set.x.shape[0]))

    def build_model(self):
        # numpy random generator
        numpy_rng = np.random.RandomState(self.numpy_rng_seed)

        logging.info('... building the model')
        # construct the Deep Belief Network
        self.dbn = DBN.DBN(
            numpy_rng=numpy_rng,
            n_ins=self.num_features,
            hidden_layers_sizes=self.hidden_layers_sizes,
            n_outs=self.num_classes
        )

    def pretrain(self):
        # start-snippet-2
        #########################
        # PRETRAINING THE MODEL #
        #########################
        logging.info('... getting the pretraining functions')
        pretraining_fns = self.dbn.pretraining_functions(
            train_set_x=self.th_train_set.x,
            batch_size=self.batch_size,
            k=self.k
        )

        logging.info('... pre-training the model')
        start_time = timeit.default_timer()
        mega_batch_size = self.num_samples / self.num_mega_batches
        # Pre-train layer-wise
        for layer_idx in xrange(self.dbn.n_layers):
            logging.debug('pretrain layer {}/{}'.format(layer_idx + 1, self.dbn.n_layers))
            # go through pretraining epochs
            for epoch in xrange(self.pretraining_epochs):
                logging.debug('pretrain layer {}, epoch {}/{}'.format(
                    layer_idx + 1, epoch + 1, self.pretraining_epochs))
                # go through the training set
                cc = np.zeros(self.num_minibatches_in_mega * self.num_mega_batches)
                cc_idx = 0

                for mega_batch_index in xrange(self.num_mega_batches):
                    lo, hi = mega_batch_index * mega_batch_size, (mega_batch_index + 1) * mega_batch_size
                    logging.debug('Setting train_set.x[{}:{}]'.format(lo, hi))
                    self.th_train_set.x.set_value(self.mm_train_set.x[lo:hi])

                    for batch_index in xrange(self.num_minibatches_in_mega):
                        logging.debug('pretrain layer {}, epoch {}, mega_batch {}/{}, batch {}/{}'.format(
                            layer_idx + 1, epoch + 1, mega_batch_index + 1, self.num_mega_batches,
                            batch_index + 1, self.num_minibatches_in_mega))
                        cost = pretraining_fns[layer_idx](index=batch_index, lr=self.pretrain_lr)
                        cc[cc_idx] = cost
                        cc_idx += 1

                        logging.debug2('W={}, hbias={}, vbias={}'.format(
                            self.dbn.rbm_layers[0].W[:2, :2].eval(), self.dbn.rbm_layers[0].hbias[:4].eval(),
                            self.dbn.rbm_layers[0].vbias[:4].eval()))

                logging.debug2('cost: {}'.format(cc))
                logging.info('Pre-training layer {:d}, epoch {:d}, avg cost {}'.format(layer_idx, epoch, np.mean(cc)))

        end_time = timeit.default_timer()
        # end-snippet-2
        logging.info('The pretraining code ran for {:.2f}m'.format((end_time - start_time) / 60.))

        if self.pretrain_model_file:
            logging.info('saving pretrain model file to {}'.format(self.pretrain_model_file))
            self.dbn.save_model(self.pretrain_model_file)
        else:
            logging.info('pretrain model not saved')

    def finetune(self):
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model
        logging.info('... getting the finetuning functions')
        # assert self.valid_size + self.test_size == self.num_samples / self.num_mega_batches

        #
        datasets = [
            (self.th_train_set.x, self.th_train_set.y),
            # (self.th_train_set.x[:self.valid_size], self.th_train_set.y[:self.valid_size]),
            # (self.th_train_set.x[self.valid_size:], self.th_train_set.y[self.valid_size:])
        ]

        mega_batch_size = self.num_samples / self.num_mega_batches
        logging.info('mega_batch_size={}'.format(mega_batch_size))
        # num_valid_batches = self.valid_size / self.batch_size   ## @todo
        # num_test_batches = self.test_size / self.batch_size

        train_fn, train_score, indices = self.dbn.build_finetune_functions2(
            datasets=datasets,
            mini_batch_size=self.batch_size,
            mega_batch_size=mega_batch_size,
            learning_rate=self.finetune_lr
        )

        logging.info('... finetuning the model')
        # early-stopping parameters
        patience = 4 * self.num_minibatches  # look as this many examples regardless
        patience_increase = 2.    # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        # go through this many minibatches before checking the network on the validation set;
        # in this case we check every epoch
        validation_frequency = min(self.num_minibatches, patience / 2)
        logging.info('patience={}, patience_increase={}, improvement_threshold={:.4f}, validation_frequency={}'.format(
            patience, patience_increase, improvement_threshold, validation_frequency))

        best_validation_loss = np.inf
        avg_test_loss = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        best_train_loss = np.inf

        logging.debug('W[:2, :4]={}, b[:4]={}'.format(
            self.dbn.logLayer.W[:2, :4].eval(), self.dbn.logLayer.b[:4].eval()))
        while (epoch < self.finetune_training_epochs) and (not done_looping):
            epoch = epoch + 1
            logging.debug('finetune epoch {}/{}'.format(epoch, self.finetune_training_epochs))
            train_loss = []

            for mega_batch_index in xrange(self.num_mega_batches):
                lo, hi = mega_batch_index * mega_batch_size, (mega_batch_index + 1) * mega_batch_size
                logging.debug('Setting train_set[{}:{}]'.format(lo, hi))
                self.th_train_set.x.set_value(self.mm_train_set.x[lo:hi])
                self.th_train_set.yf.set_value(self.mm_train_set.y[lo:hi])

                for minibatch_index in xrange(self.num_minibatches_in_mega):
                    logging.debug('finetune epoch {}, megabatch {}/{}, minibatch {}/{}'.format(
                        epoch, mega_batch_index + 1, self.num_mega_batches,
                        minibatch_index + 1, self.num_minibatches_in_mega))
                    # ****** execute the update ******
                    logging.debug2('x_begin/end, y_begin/end={}'.format(indices(minibatch_index)))
                    minibatch_avg_cost = train_fn(minibatch_index)
                    logging.debug2('hiddenLayer0 W[:1, :4]={}, b[:4]={}'.format(
                        self.dbn.sigmoid_layers[0].W[:1, :4].eval(), self.dbn.sigmoid_layers[0].b[:4].eval()))
                    logging.debug2('logLayer W[:1, :4]={}, b[:4]={}'.format(
                        self.dbn.logLayer.W[:1, :4].eval(), self.dbn.logLayer.b[:4].eval()))
                    # ********************************
                    logging.info('minibatch_avg_cost={}'.format(minibatch_avg_cost))
                    # iter_num = (epoch - 1) * self.num_minibatches_in_mega + minibatch_index

                    # if (iter_num + 1) % validation_frequency == 0:
                    #     logging.debug('VALIDATION finetune epoch {}, megabatch {}, minibatch {}, iter_num {}'.format(
                    #         epoch, mega_batch_index, minibatch_index, iter_num))
                    #     import pdb; pdb.set_trace()
                    #     validation_losses = validate_score(0, self.num_minibatches_in_mega - 1, mega_batch_index)
                    #     this_validation_loss = np.mean(validation_losses)
                    #     logging.info('epoch {:d}, minibatch {:d}/{:d}, validation error {:f} %'.format(
                    #         epoch, minibatch_index + 1, self.num_minibatches_in_mega, this_validation_loss * 100.))

                    #     # if we got the best validation score until now
                    #     if this_validation_loss < best_validation_loss:

                    #         # improve patience if loss improvement is good enough
                    #         if (
                    #             this_validation_loss < best_validation_loss *
                    #             improvement_threshold
                    #         ):
                    #             patience = max(patience, iter_num * patience_increase)

                    #         # save best validation score and iteration number
                    #         best_validation_loss = this_validation_loss
                    #         best_iter = iter_num

                    #         # test it on the test set
                    #         test_losses = test_score(num_test_batches)
                    #         avg_test_loss = np.mean(test_losses)
                    #         logging.info('     epoch {:d}, minibatch {:d}/{:d}, test error of best model {} %'.format(
                    #             epoch, minibatch_index + 1, self.num_minibatches_in_mega, avg_test_loss * 100.))

                    # if patience <= iter_num:
                    #     done_looping = True
                    #     break

                train_loss.extend(train_score(0, self.num_minibatches_in_mega))
                logging.debug2('train_loss={}'.format(train_loss))

            avg_train_loss = np.mean(train_loss)
            logging.info('avg_train_loss={}'.format(avg_train_loss))
            if avg_train_loss < best_train_loss:
                if avg_train_loss < best_train_loss * improvement_threshold:
                    patience = max(patience, epoch * patience_increase)
                best_train_loss = avg_train_loss

            if patience <= epoch:
                done_looping = True
                break

        end_time = timeit.default_timer()
        logging.info(
            (
                'Optimization complete with avg train error of %f %%, best validation error of %f %%, '
                'obtained at iteration %i, with avg test error %f %%'
            ) % (avg_train_loss * 100., best_validation_loss * 100., epoch, avg_test_loss * 100.)
        )
        logging.info('The fine tuning code ran for {:.2f}m'.format((end_time - start_time) / 60.))

        logging.info('Saving finetuned model file to {}'.format(self.finetuned_model_file))
        self.dbn.save_model(self.finetuned_model_file)
        logging.info('done saving')
