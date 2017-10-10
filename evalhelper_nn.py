"""Evaluation helper - Neural Network."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import random
import pdb

# Third-party libraries
import numpy as np
import math

# ---------------------------------------------------------------------------
# Class and functions
# ---------------------------------------------------------------------------

class Network(object):

    def __init__(self, sizes):
        """ Constructor.

        Parameters
        ----------
        sizes : list
            layers and neutrons

        Returns
        -------
        None : none
            random initial weights and bias

        """

        # layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        # random initials with fixed seed
        random.seed(1)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [(np.random.randn(x, y)/math.sqrt(self.sizes[0])) for x, y in zip(self.sizes[1:], self.sizes[:-1])]
        # batch, iterations and learning rate
        self.mini_batchsize = 50
        self.epochs = 3
        self.eta = 0.1


    def SGD(self, training_data, test_data=None):
        """ Stochastic gradient descent.

        Parameters
        ----------
        training data : list of tuples
            data to train the NN

        test data : list of tuples
            optional test data to evaluate NN after each epoch

        Returns
        -------
        None : none

        """

        # length of training data
        n = len(training_data)

        # length of optional test data
        if test_data: n_test = len(test_data)

        # loop throug epochs
        for j in range(self.epochs):
        	# randomly sort training data for mini batches
            random.shuffle(training_data)
            # one mini batch only -> seems to be described in exercise (point c)
            #self.update_mini_batch(training_data[0:self.mini_batchsize], self.eta)
            # create mini batches -> use entire training_data set
            mini_batches = [ training_data[k:k+self.mini_batchsize] for k in range(0, n, self.mini_batchsize)]
            # loop through mini batches and update weights
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.eta)
            # print progress if test data is provided
            if test_data:
                evalu = self.evaluate(test_data)
                print("Epoch:", (j+1), evalu, "from" , n_test, "->", round(evalu/n_test*100,2), "%")
            else:
                print("Epoch complete: ", j)


    def update_mini_batch(self, mini_batch, eta):
        """ Upddate weights for a single mini batch.

        Parameters
        ----------
        mini_batch : list of tuples
            tuples that are to be "evaluated" for the SDG update

        eta : float
            learning rate

        Returns
        -------
        None : none
            update of weights and bias of NN object

        """

        # storage for nablas
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # derivative for single training pair -> use general backprob for gradient (!)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # elemtwise addition (!)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weights
        self.weights = [w-(float(eta)/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(float(eta)/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """ Partial derivative of cost function for single pair x,y.

        Parameters
        ----------
        x : ndarray
            box

        y : ndarray
            result (crossed vs uncrossed)

        Returns
        -------
        (nabla_b, nabla_w) : tuple of floats
            gradient of single pair

        """

        # storage
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        # loop through the NN and save a,z for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # delta^L -> delta of output layer -> BP1
        delta = self.delta_ce(activations[-1], y)
        # partial derivs after b and W for last layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # backward calculation for all deltas and respective derivatives
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # BP2
            delta = np.dot(self.weights[-l+1].transpose(), delta) * float(sp)
            # BP3
            nabla_b[-l] = delta
            # BP4
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # return gradients
        return (nabla_b, nabla_w)


    def delta_ce(self, a, y):
        """ Delta for cross-entropy cost function.

        Parameters
        ----------
        a : float
            cost input

        y : float
            result

        Returns
        -------
        (a-y) : float
            delta

        """

        return (a-y)


    def feedforward(self, a):
        """ Feedforward of NN.

        Parameters
        ----------
        a : ndarray
            input box

        Returns
        -------
        a : ndarray
            predicted NN result

        """

        # loop through NN
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        # return prediction
        return a


    def evaluate(self, test_data):
        """ Evaluate the accuracy of the NN given test_data.

        Parameters
        ----------
        test_data : list of tuples
            data to be tested with the NN

        Returns
        -------
        result : float
            accuracy of NN, number of correct classifications

        """

        # evaluate each x of test data
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        # return sum of correct classifications
        return sum(int(x == y) for (x, y) in test_results)


    def crossed(self, inputdata):
        """ Check if single box is crossed.

        Parameters
        ----------
        inputdata : ndarray
            input box

        Returns
        -------
        result : ndarray
            (0,1) or (1,0) vector

        """

        return (np.argmax(self.feedforward(inputdata)))


#####
# General functions
def sigmoid(z):
    """ Sigmoid activation function.

    Parameters
    ----------
    z : ndarray
        vector to be activated

    Returns
    -------
    result : ndarray
        activated vector

    """

    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """ Derivative of the sigmoid activation function.

    Parameters
    ----------
    z : ndarray
        point at which derivative is evaluated

    Returns
    -------
    result : ndarray
        derivative at z

    """

    return sigmoid(z)*(1-sigmoid(z))
