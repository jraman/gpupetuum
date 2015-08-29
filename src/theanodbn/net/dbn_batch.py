'''
Run DBN by loading minibatches into GPU memory.
Needed when all of the data does not fit into GPU memory.

Notes:
 * validation and test sets overlap with training set.
 * We use parts of the last megabatch as validation and test sets.
'''

import csv
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
logging.DEBUG2 = DEBUG2_LEVEL_NUM


class FeatureSet(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.yf = None


class LoadFrom(object):
    RAM = 'RAM'
    disk = 'disk'


class DiskLoader(object):
    def __init__(self, dataset_list_file):
        assert os.path.exists(dataset_list_file), 'Cannot find file {}'.format(dataset_list_file)
        self._dataset_list_file = dataset_list_file
        self._files = None
        with open(self._dataset_list_file) as fobj:
            reader = csv.reader(fobj)
            self._files = [row for row in reader if row and not row[0].startswith('#')]

        for row in self._files:
            assert len(row) == 2, 'Expected 2 filenames.  Got {}'.format(row)
            datafile, labelfile = row
            assert os.path.exists(datafile), 'Cannot find file {}'.format(datafile)
            assert os.path.exists(labelfile), 'Cannot find file {}'.format(labelfile)

        self._datafiles = [r[0] for r in self._files]
        self._labelfiles = [r[1] for r in self._files]

    @property
    def datafiles(self):
        return self._datafiles

    @property
    def labelfiles(self):
        return self._labelfiles


class DbnMegaBatch(object):
    def __init__(
        self,
        dataset_file,
        label_file,
        pretrain_model_file,
        finetuned_model_file,
        load_from='RAM',
        hidden_layers_sizes=[1024],
        pretraining_epochs=100,
        pretrain_lr=0.01,
        k=1,
        finetune_training_epochs=1000,
        finetune_lr=0.1,
        dataset_dir=None,
        num_mega_batches=None,
        batch_size=10,
        numpy_rng_seed=4242,
        valid_size=None,
        test_size=None,
    ):
        """
        Either:
         * Load to RAM and transfer as neeeded to GPU/CPU (shared variable), or
            - specify num_mega_batches
         * Load from disk and transfer to GPU/CPU (shared variable).
            - dataset_file is a text file listing tuples of (binary dataset_file, label_file)
            - label_file is None

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
        assert load_from in (LoadFrom.RAM, LoadFrom.disk), 'Can only load from {} or {}, not {}'.format(
            LoadFrom.RAM, LoadFrom.disk, load_from)

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
        self.load_from = load_from
        self.hidden_layers_sizes = hidden_layers_sizes
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.k = k
        self.finetune_training_epochs = finetune_training_epochs
        self.finetune_lr = finetune_lr
        self.num_mega_batches = num_mega_batches
        self.batch_size = batch_size
        self.numpy_rng_seed = numpy_rng_seed
        self.valid_size = valid_size or 0
        self.test_size = test_size or 0

        self.num_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.num_minibatches = 0                # across all mega
        self.num_minibatches_in_mega = 0        # num batches in a mega batch
        self.mm_train_set = FeatureSet()

        # logging level is expected to be set *before* this class is instantiated
        self.global_logging_level = logging.root.getEffectiveLevel()

        if load_from == LoadFrom.disk:
            self.disk_loader = DiskLoader(dataset_file)
        else:
            self.disk_loader = None

        self.dbn = None

    def run(self):
        logging.info('THEANO_FLAGS={}'.format(os.getenv('THEANO_FLAGS')))
        if self.load_from == LoadFrom.RAM:
            self.load_data_mm()
        else:
            self.prepare()
        self.build_model()
        self.setup_shared_xy()
        self.pretrain()
        self.finetune()

    def prepare(self):
        '''Set num_samples, num_mega_batches, num_features
        Read one file and set num_features, etc.
        This is a bit underoptimized.  We end up loading the first set of files twice - once here and
        once at start of pretraining.
        '''
        with open(self.disk_loader.datafiles[0], 'rb') as fobj:
            self.num_features = len(nn_util.load_pickle_file(fobj).next())
            self.num_samples = 1 + sum(1 for x in nn_util.load_pickle_file(fobj))
        self.num_minibatches = len(self.disk_loader.datafiles)
        self.num_samples *= self.num_minibatches
        with open(self.label_file) as fobj:
            labels = [int(x.strip()) for x in fobj]
        self.num_classes = len(labels)
        self._label2idx = dict([(ll, ii) for ii, ll in enumerate(sorted(labels))])
        logging.info('num_samples={}, num_features={}, num_classes={}'.format(
            self.num_samples, self.num_features, self.num_classes))

        with open(self.disk_loader.labelfiles[0]) as fobj:
            num_y_samples = sum(1 for x in fobj)
        num_y_samples *= len(self.disk_loader.labelfiles)
        assert self.num_samples == num_y_samples, "Num rows in X and Y don't match"

        self.num_minibatches = self.num_samples / self.batch_size
        self.num_minibatches_in_mega = self.num_samples / self.batch_size / self.num_mega_batches
        logging.info('num_minibatches={}, num_minibatches_in_mega={}'.format(
            self.num_minibatches, self.num_minibatches_in_mega))

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
            label2idx = dict((ll, ii) for ii, ll in enumerate(sorted(unique_labels)))
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

    def setup_shared_xy(self, borrow=True):
        '''Set up theano shared variables for input X and output y.
        X is a matrix of num_samples_per_mega_batch x num_features
        y is a vector of num_samples_per_mega_batch x 1
        '''
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

        if self.global_logging_level <= logging.DEBUG2:
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

        logging.info('Param shapes: {}'.format(
            ', '.join(['{}:{}'.format(p, p.shape.eval()) for p in self.dbn.params])))

    def load_shared_from_ram(self, mega_batch_index, load_y):
        '''Load data from main memory to theano shared variable.
        '''
        mega_batch_size = self.num_samples / self.num_mega_batches
        lo, hi = mega_batch_index * mega_batch_size, (mega_batch_index + 1) * mega_batch_size
        if self.global_logging_level <= logging.DEBUG2:
            logging.debug2('Setting train_set.x[{}:{}]'.format(lo, hi))
        self.th_train_set.x.set_value(self.mm_train_set.x[lo:hi])
        if load_y:
            self.th_train_set.yf.set_value(self.mm_train_set.y[lo:hi])

    def load_shared_from_disk(self, mega_batch_index, load_y):
        datafile = self.disk_loader.datafiles[mega_batch_index]
        with open(datafile, 'rb') as fobj:
            xx = np.array([x for x in nn_util.load_pickle_file(fobj)], dtype=theano.config.floatX)
        if self.global_logging_level <= logging.DEBUG:
            logging.debug('Loaded {}'.format(datafile))

        self.th_train_set.x.set_value(xx)

        if load_y:
            labelfile = self.disk_loader.labelfiles[mega_batch_index]
            with open(labelfile, 'r') as ff:
                yy = np.array([self._label2idx[int(line.strip())] for line in ff], dtype=theano.config.floatX)
            if self.global_logging_level <= logging.DEBUG:
                logging.debug('Loaded {}'.format(labelfile))

            self.th_train_set.yf.set_value(yy)

    def load_mega_batch(self, mega_batch_index, load_y=False):
        if self.load_from == LoadFrom.RAM:
            self.load_shared_from_ram(mega_batch_index, load_y)
        else:
            self.load_shared_from_disk(mega_batch_index, load_y)

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

        # special case optimization: load only once if num_mega_batches == 1
        if self.num_mega_batches == 1:
            logging.info('Single megabatch.  Loading it into GPU/CPU')
            self.load_mega_batch(0, load_y=False)

        # Pre-train layer-wise
        for layer_idx in xrange(self.dbn.n_layers):
            logging.info('pretrain layer {}/{}'.format(layer_idx + 1, self.dbn.n_layers))
            # go through pretraining epochs
            for epoch in xrange(self.pretraining_epochs):
                logging.info('pretrain layer {}, epoch {}/{}'.format(
                    layer_idx + 1, epoch + 1, self.pretraining_epochs))
                # go through the training set
                cc = np.zeros(self.num_minibatches_in_mega * self.num_mega_batches)
                cc_idx = 0

                for mega_batch_index in xrange(self.num_mega_batches):
                    if self.num_mega_batches > 1:
                        self.load_mega_batch(mega_batch_index, load_y=False)

                    for batch_index in xrange(self.num_minibatches_in_mega):
                        if self.global_logging_level <= logging.DEBUG:
                            logging.debug('pretrain layer {}, epoch {}, mega_batch {}/{}, batch {}/{}'.format(
                                layer_idx + 1, epoch + 1, mega_batch_index + 1, self.num_mega_batches,
                                batch_index + 1, self.num_minibatches_in_mega))
                        cost = pretraining_fns[layer_idx](index=batch_index, lr=self.pretrain_lr)
                        cc[cc_idx] = cost
                        cc_idx += 1

                        if self.global_logging_level <= logging.DEBUG2:
                            logging.debug2('W={}, hbias={}, vbias={}'.format(
                                self.dbn.rbm_layers[0].W[:2, :2].eval(), self.dbn.rbm_layers[0].hbias[:4].eval(),
                                self.dbn.rbm_layers[0].vbias[:4].eval()))

                if self.global_logging_level <= logging.DEBUG2:
                    logging.debug2('cost: {}'.format(cc))
                logging.info('pre-train layer {:d}, epoch {:d}, avg cost {}'.format(layer_idx, epoch, np.mean(cc)))

                l2 = [p.norm(2).eval() for p in self.dbn.params]
                l2all = np.sqrt(sum(x * x for x in l2))
                logging.info('pre-train layer {:d}, epoch {}, L2 norms: all={}, individual={}'.format(
                    layer_idx, epoch, l2all, ', '.join(str(x) for x in l2)))

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

        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        best_train_loss = np.inf

        if self.global_logging_level <= logging.DEBUG2:
            logging.debug2('W[:2, :4]={}, b[:4]={}'.format(
                self.dbn.logLayer.W[:2, :4].eval(), self.dbn.logLayer.b[:4].eval()))

        # special case optimization: load only once if num_mega_batches == 1
        if self.num_mega_batches == 1:
            logging.info('Single megabatch.  Loading it into GPU/CPU')
            self.load_mega_batch(0, load_y=True)

        while (epoch < self.finetune_training_epochs) and (not done_looping):
            learning_rate = self.finetune_lr / np.sqrt(epoch + 1)
            learning_rate = np.asscalar(np.array(learning_rate, dtype=theano.config.floatX))
            epoch = epoch + 1
            logging.info('finetune epoch {}/{}, learning_rate={:.4f}'.format(
                epoch, self.finetune_training_epochs, learning_rate))
            train_loss = []

            for mega_batch_index in xrange(self.num_mega_batches):
                if self.num_mega_batches > 1:
                    self.load_mega_batch(mega_batch_index, load_y=True)

                for minibatch_index in xrange(self.num_minibatches_in_mega):
                    if self.global_logging_level <= logging.DEBUG:
                        logging.debug('finetune epoch {}, megabatch {}/{}, minibatch {}/{}'.format(
                            epoch, mega_batch_index + 1, self.num_mega_batches,
                            minibatch_index + 1, self.num_minibatches_in_mega))
                    # ****** execute the update ******
                    if self.global_logging_level <= logging.DEBUG2:
                        logging.debug2('x_begin/end, y_begin/end={}'.format(indices(minibatch_index)))
                    minibatch_avg_cost = train_fn(minibatch_index, learning_rate)
                    if self.global_logging_level <= logging.DEBUG2:
                        logging.debug2('hiddenLayer0 W[:1, :4]={}, b[:4]={}'.format(
                            self.dbn.sigmoid_layers[0].W[:1, :4].eval(), self.dbn.sigmoid_layers[0].b[:4].eval()))
                    if self.global_logging_level <= logging.DEBUG2:
                        logging.debug2('logLayer W[:1, :4]={}, b[:4]={}'.format(
                            self.dbn.logLayer.W[:1, :4].eval(), self.dbn.logLayer.b[:4].eval()))
                    # ********************************
                    logging.info('minibatch_avg_cost={}'.format(minibatch_avg_cost))

                train_loss.extend(train_score(0, self.num_minibatches_in_mega))
                if self.global_logging_level <= logging.DEBUG2:
                    logging.debug2('train_loss={}'.format(train_loss))

            l2 = [p.norm(2).eval() for p in self.dbn.params]
            l2all = np.sqrt(sum(x * x for x in l2))
            logging.info('epoch {}: L2 norms: all={}, individual={}'.format(
                epoch, l2all, ', '.join(str(x) for x in l2)))

            avg_train_loss = np.mean(train_loss)
            logging.info('epoch {}: avg_train_loss={}'.format(epoch, avg_train_loss))
            if avg_train_loss < best_train_loss:
                if avg_train_loss < best_train_loss * improvement_threshold:
                    patience = max(patience, epoch * patience_increase)
                best_train_loss = avg_train_loss

            if patience <= epoch:
                done_looping = True
                break

        end_time = timeit.default_timer()
        logging.info('Optimization complete with avg train error of {:f}'.format(avg_train_loss))
        logging.info('The fine tuning code ran for {:.2f}m'.format((end_time - start_time) / 60.))

        logging.info('Saving finetuned model file to {}'.format(self.finetuned_model_file))
        self.dbn.save_model(self.finetuned_model_file)
        logging.info('done saving')
