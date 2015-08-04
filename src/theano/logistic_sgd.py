"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import os
import sys
import timeit

import logging
import numpy as np

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, include_bias=True):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type include_bias: boolean
        :param include_bias: add bias term.  By default bias term is excluded.
        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        if include_bias:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        if include_bias:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        else:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        # self.params = [self.W, self.b]
        self.params = [self.W, ]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError('can handle only int y values at this time')


def _shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_pickle_file(fobj):
    while True:
        try:
            yield cPickle.load(fobj)
        except EOFError:
            break


def load_data(dataset, label_file):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset

    # train_set is a tuple(ndarray1, ndarray2)
    # ndarray1 and ndarray2 are of length = num_samples
    # each elem of ndarray1 is an ndarray of length = num_features (dtype=float32)
    # ndarray2 is of length num_samples, each element is the image label (an int32)

    For imnet, we set test = valid = train
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    with open(dataset, 'rb') as fobj:
        train_set_x = np.array([x for x in load_pickle_file(fobj)])

    with open(label_file, 'r') as ff:
        train_set_label = np.array([int(line.strip()) for line in ff], dtype=np.int32)

    # theano requires that the labels be in the range [0, L), where L is the number of unique labels.
    label2idx = dict((ll, ii) for ii, ll in enumerate(set(train_set_label)))
    train_set_y = np.array([label2idx[ll] for ll in train_set_label], dtype=np.int32)

    nrow = train_set_x.shape[0]
    assert nrow == train_set_y.shape[0], "Num rows in X and Y don't match"

    # with replacement
    np.random.seed(4242)
    valid_set_idx = np.random.choice(nrow, 0.3 * nrow)
    valid_set_x, valid_set_y = train_set_x[valid_set_idx], train_set_y[valid_set_idx]
    logging.info('Selected {} rows as validation set'.format(len(valid_set_idx)))

    np.random.seed(4343)
    test_set_idx = np.random.choice(nrow, 0.2 * nrow)
    test_set_x, test_set_y = train_set_x[test_set_idx], train_set_y[test_set_idx]
    logging.info('Selected {} rows as test set'.format(len(test_set_idx)))

    # train_set, valid_set, test_set format: tuple(input, target)
    # input, x, is an np.ndarray of 2 dimensions (a matrix)
    # with each row corresponding to an example. target, y, is a
    # np.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. The target should be associated
    # with the same index in the input.

    shared_train_set_x, shared_train_set_y = _shared_dataset(train_set_x, train_set_y)
    shared_valid_set_x, shared_valid_set_y = _shared_dataset(valid_set_x, valid_set_y)
    shared_test_set_x, shared_test_set_y = _shared_dataset(test_set_x, test_set_y)

    rval = [
        (shared_train_set_x, shared_train_set_y),
        (shared_valid_set_x, shared_valid_set_y),
        (shared_test_set_x, shared_test_set_y),
    ]
    return rval


def sgd_optimization_mnist(
    learning_rate=0.13,
    n_epochs=1000,
    dataset='../imnet_data/imnet_sample.n3.pkl',
    label_file='../imnet_data/label_sample.n3.txt',
    batch_size_min=600,
    best_model='best_model.pkl',
    include_bias=True,
):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    logging.info('Loading dataset from {}'.format(dataset))
    datasets = load_data(dataset, label_file)
    logging.info('Done loading dataset')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    num_features = train_set_x.get_value(borrow=True).shape[0]
    num_samples = train_set_x.get_value(borrow=True).shape[1]
    num_classes = len(set(train_set_y.owner.inputs[0].get_value(borrow=True)))
    logging.info('num_features={}, num_samples={}, num_classes={}'.format(num_features, num_samples, num_classes))

    # compute number of minibatches for training, validation and testing
    batch_size_train = min(batch_size_min, num_features)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size_train
    logging.info('train batch_size={}, num_batches={}'.format(batch_size_train, n_train_batches))
    batch_size_valid = min(batch_size_min, valid_set_x.get_value(borrow=True).shape[0])
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size_valid
    logging.info('validate batch_size={}, num_batches={}'.format(batch_size_valid, n_valid_batches))
    batch_size_test = min(batch_size_min, test_set_x.get_value(borrow=True).shape[0])
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size_test
    logging.info('test batch_size={}, num_batches={}'.format(batch_size_test, n_test_batches))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    logging.info('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=num_samples, n_out=num_classes, include_bias=include_bias)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    get_test_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size_test: (index + 1) * batch_size_test],
            y: test_set_y[index * batch_size_test: (index + 1) * batch_size_test]
        }
    )

    get_validate_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size_valid: (index + 1) * batch_size_valid],
            y: valid_set_y[index * batch_size_valid: (index + 1) * batch_size_valid]
        }
    )

    get_train_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size_train: (index + 1) * batch_size_train],
            y: train_set_y[index * batch_size_train: (index + 1) * batch_size_train]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_w = T.grad(cost=cost, wrt=classifier.W)
    if include_bias:
        g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_w)]
    if include_bias:
        updates.append((classifier.b, classifier.b - learning_rate * g_b))

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size_train: (index + 1) * batch_size_train],
            y: train_set_y[index * batch_size_train: (index + 1) * batch_size_train]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    logging.info('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    # go through this many minibatches before checking the network
    # on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            logging.info('minibatch_avg_cost={}'.format(minibatch_avg_cost))
            # iteration number
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [get_validate_error(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                logging.info(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        logging.info('increasing patience from {} to {}'.format(patience, iter_num * patience_increase))
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [get_test_error(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    logging.info(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open(best_model, 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter_num:
                done_looping = True
                break

    end_time = timeit.default_timer()

    train_errors = [get_train_error(i) for i in xrange(n_train_batches)]
    avg_train_error = np.mean(train_errors)

    print(
        (
            'Optimization complete with best validation error of %.4f %%,'
            ' test error of %.4f %%, and train error of %.4f %%'
        )
        % (best_validation_loss * 100., test_score * 100., avg_train_error * 100.0)
    )
    total_time = end_time - start_time
    logging.info('Completed {} epochs, at {:.4f} epochs/sec.  Total time {:.2f}s'.format(
        epoch, 1. * epoch / total_time, total_time))


def predict(dataset='mnist.pkl.gz', modelfile='best_model.pkl'):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open(modelfile))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values


def main():
    log_format = '%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s'
    logging.basicConfig(format=log_format, level=logging.DEBUG)
    logging.info('Running {}'.format(os.path.abspath(__file__)))
    logging.info('sys.argv: {}'.format(sys.argv))
    sgd_optimization_mnist(include_bias=False, n_epochs=100)


if __name__ == '__main__':
    main()
