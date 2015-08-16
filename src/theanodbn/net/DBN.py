"""
"""
import cPickle
import logging
import numpy
import os
import timeit

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM


# Add another level to the logger
DEBUG2_LEVEL_NUM = 9
logging.addLevelName(DEBUG2_LEVEL_NUM, "DEBUG2")


def debug2(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(DEBUG2_LEVEL_NUM):
        self._log(DEBUG2_LEVEL_NUM, message, args, **kws)
logging.Logger.debug2 = debug2
logging.debug2 = logging.root.debug2


# start-snippet-1
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        logging.info('Number of hidden layers, n_layers={}'.format(self.n_layers))

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input_x=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None,
                                                 k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = self.get_num_batches(train_set_x, batch_size)
        n_valid_batches = self.get_num_batches(valid_set_x, batch_size)
        n_test_batches = self.get_num_batches(test_set_x, batch_size)
        logging.info('batch sizes: train={}, valid={}, test={}'.format(
            n_train_batches, n_valid_batches, n_test_batches))

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        train_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # Create a function that scans the entire training set
        def train_score():
            return [train_score_i(i) for i in xrange(n_train_batches)]

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, train_score, valid_score, test_score

    def build_finetune_functions2(self, datasets, mini_batch_size, mega_batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type mini_batch_size: int
        :param mini_batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        assume all of y is loaded into GPU memory, but x is loaded in megabatches.
        '''
        assert mini_batch_size, 'mini_batch_size cannot be zero'

        (train_set_x, train_set_y) = datasets[0]
        # (valid_set_x, valid_set_y) = datasets[1]
        # (test_set_x, test_set_y) = datasets[2]

        mini_index = T.lscalar('mini_index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        # updates = [(param, param - gparam * learning_rate)
        #            for param, gparam in zip(self.params, gparams)]
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        x_begin = mini_index * mini_batch_size
        x_end = x_begin + mini_batch_size

        y_begin = mini_index * mini_batch_size
        y_end = y_begin + mini_batch_size

        train_fn = theano.function(
            inputs=[mini_index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[x_begin:x_end],
                self.y: train_set_y[y_begin:y_end]
            }
        )

        train_score_i = theano.function(
            inputs=[mini_index],
            outputs=self.errors,
            givens={
                self.x: train_set_x[x_begin:x_end],
                self.y: train_set_y[y_begin:y_end]
            }
        )

        # valid_score_i = theano.function(
        #     inputs=[mini_index, mega_index],
        #     outputs=self.errors,
        #     givens={
        #         self.x: valid_set_x[x_begin:x_end],
        #         self.y: valid_set_y[y_begin:y_end]
        #     }
        # )

        # test_score_i = theano.function(
        #     inputs=[mini_index, mega_index],
        #     outputs=self.errors,
        #     givens={
        #         self.x: test_set_x[x_begin:x_end],
        #         self.y: test_set_y[y_begin:y_end]
        #     }
        # )

        # Create a function that scans the entire training set
        def train_score(mini_idx_begin, mini_idx_end):
            '''inclusive of mini_idx_begin and exclusive of mini_idx_end'''
            logging.debug2('mini_idx_begin={}, end={}'.format(mini_idx_begin, mini_idx_end))
            return [train_score_i(i) for i in xrange(mini_idx_begin, mini_idx_end)]

        f1 = theano.function(
            inputs=[mini_index],
            outputs=[x_begin, x_end, y_begin, y_end]
        )

        # Create a function that scans the entire validation set
        # def valid_score(mini_idx_begin, mini_idx_end, mega_index):
        #     return [valid_score_i(i, mega_index) for i in xrange(mini_idx_begin, mini_idx_end + 1)]

        # Create a function that scans the entire test set
        # def test_score(n_test_batches):
        #     return [test_score_i(i) for i in xrange(n_test_batches)]

        # return train_fn, train_score, valid_score, test_score
        return train_fn, train_score, f1

    def get_num_batches(self, shared_set_x, batch_size):
        '''Given a shared variable, shared_set_x, and batch_size, return the number of batches.
        '''
        num_batches = int(numpy.ceil(shared_set_x.get_value(borrow=True).shape[0] / float(batch_size)))
        return num_batches

    def save_model(self, filename):
        with open(filename, 'w') as ff:
            cPickle.dump(self, ff)

    @staticmethod
    def load_model(filename):
        with open(filename, 'r') as ff:
            return cPickle.load(ff)


def test_dbn(
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
    logging.info('THEANO_FLAGS={}'.format(os.getenv('THEANO_FLAGS')))
    datasets = load_data(dataset_file, label_file, valid_size=valid_size, test_size=test_size)

    train_set_x, train_set_y = datasets[0]
    num_features = train_set_x.get_value(borrow=True).shape[1]
    num_classes = len(set(train_set_y.eval()))
    logging.info('num_features={}, num_classes={}'.format(num_features, num_classes))

    logging.info('hidden_layers_sizes={}'.format(hidden_layers_sizes))
    logging.info('pretraining_epochs={}'.format(pretraining_epochs))
    logging.info('pretrain_lr={}'.format(pretrain_lr))
    logging.info('CD-k={}'.format(k))
    logging.info('finetune_training_epochs={}'.format(finetune_training_epochs))
    logging.info('finetune_lr={}'.format(finetune_lr))
    logging.info('batch_size={}'.format(batch_size))
    logging.info('numpy_rng seed={}'.format(numpy_rng_seed))

    # numpy random generator
    numpy_rng = numpy.random.RandomState(numpy_rng_seed)

    logging.info('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(
        numpy_rng=numpy_rng,
        n_ins=num_features,
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=num_classes
    )

    # compute number of minibatches for training, validation and testing
    n_train_batches = dbn.get_num_batches(train_set_x, batch_size)
    logging.info('n_train_batches={}'.format(n_train_batches))

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    logging.info('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    logging.info('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for ii in xrange(dbn.n_layers):
        logging.debug('pretrain layer {}/{}'.format(ii + 1, dbn.n_layers))
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            logging.debug('pretrain layer {}, epoch {}/{}'.format(ii + 1, epoch + 1, pretraining_epochs))
            # go through the training set
            cc = []
            for batch_index in xrange(n_train_batches):
                logging.debug('pretrain layer {}, epoch {}, batch {}/{}'.format(
                    ii + 1, epoch + 1, batch_index + 1, n_train_batches))
                cc.append(pretraining_fns[ii](index=batch_index,
                                              lr=pretrain_lr))
                logging.debug2('W={}, hbias={}, vbias={}'.format(
                    dbn.rbm_layers[0].W[:2, :2].eval(), dbn.rbm_layers[0].hbias[:4].eval(),
                    dbn.rbm_layers[0].vbias[:4].eval()))
            logging.debug2('cost={}'.format(cc))
            logging.info('Pre-training layer {:d}, epoch {:d}, avg cost {}'.format(ii, epoch, numpy.mean(cc)))

    end_time = timeit.default_timer()
    # end-snippet-2
    logging.info('The pretraining code ran for {:.2f}m'.format((end_time - start_time) / 60.))

    if pretrain_model_file:
        logging.info('saving pretrain model file to {}'.format(pretrain_model_file))
        dbn.save_model(pretrain_model_file)
    else:
        logging.info('not saving model file')

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    logging.info('... getting the finetuning functions')
    train_fn, train_model, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    logging.info('... finetuning the model')
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many minibatches regardless
    patience_increase = 2.    # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    # go through this many minibatches before checking the network on the validation set;
    # in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    avg_test_loss = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    logging.debug2('W[:2, :4]={}, b[:4]={}'.format(
        dbn.logLayer.W.eval()[:2, :4], dbn.logLayer.b[:4].eval()))
    while (epoch < finetune_training_epochs) and (not done_looping):
        epoch = epoch + 1
        logging.debug('finetune epoch {}/{}'.format(epoch, finetune_training_epochs))
        for minibatch_index in xrange(n_train_batches):
            logging.debug('finetune epoch {}, minibatch {}/{}'.format(epoch, minibatch_index + 1, n_train_batches))
            # ****** execute the update ******
            minibatch_avg_cost = train_fn(minibatch_index)
            logging.debug2('W={}, b={}'.format(
                           dbn.logLayer.W[:1, :4].eval(), dbn.logLayer.b[:4].eval()))
            # ********************************
            logging.info('minibatch_avg_cost={}'.format(minibatch_avg_cost))
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                logging.debug('finetune epoch {}, minibatch {}, iter_num {}'.format(
                    epoch, minibatch_index, iter_num))
                validation_losses = validate_model()
                logging.debug2('validation_losses={}'.format(validation_losses))
                this_validation_loss = numpy.mean(validation_losses)
                logging.info('epoch {:d}, minibatch {:d}/{:d}, validation error {:f} %'.format(
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter_num * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter_num

                    # test it on the test set
                    test_losses = test_model()
                    avg_test_loss = numpy.mean(test_losses)
                    logging.info('     epoch {:d}, minibatch {:d}/{:d}, test error of best model {} %'.format(
                        epoch, minibatch_index + 1, n_train_batches, avg_test_loss * 100.))

            if patience <= iter_num:
                done_looping = True
                break

        train_loss = train_model()
        logging.debug2('train_loss={}'.format(train_loss))

    avg_train_loss = numpy.mean(train_model())
    end_time = timeit.default_timer()
    logging.info(
        (
            'Optimization complete with avg train error of %f %%, best validation error of %f %%, '
            'obtained at iteration %i, with avg test error %f %%'
        ) % (avg_train_loss * 100., best_validation_loss * 100., best_iter + 1, avg_test_loss * 100.)
    )
    logging.info('The fine tuning code ran for {:.2f}m'.format((end_time - start_time) / 60.))

    logging.info('Saving finetuned model file to {}'.format(finetuned_model_file))
    dbn.save_model(finetuned_model_file)
    logging.info('done saving')


if __name__ == '__main__':
    test_dbn()
