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
        """ Constructor
        -> sizes : layers & neutrons
        -> random intial weights and bias
        """

        # layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        # random initials
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [(np.random.randn(x, y)/math.sqrt(self.sizes[0])) for x, y in zip(self.sizes[1:], self.sizes[:-1])]
        # batch, iterations and learning rate
        self.mini_batchsize = 50
        self.epochs = 3
        self.eta = 0.1


    def SGD(self, training_data, test_data=None):
        """ tbd
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
                #print("Epoch:", j, evalu, "/" , n_test, "->", round(evalu/n_test*100,2), "%")
                print(str(evalu) + " from " + str(n_test))
            else:
                print("Epoch complete: ", j)


    def update_mini_batch(self, mini_batch, eta):
        """ tbd
        """
        # storage for nablas
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # deriv for single training pair
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # elemtwise addition (!)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(float(eta)/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        #pdb.set_trace()
        self.biases = [b-(float(eta)/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """tbd 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.delta_ce(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # backward calculation for all deltas and respective derivs
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * float(sp)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def delta_ce(self, a, y):
        """ delta for cross-entropy cost function
        """
        return (a-y)


    def feedforward(self, a):
        """ output of network
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def evaluate(self, test_data):
        """ evaluates number of correct classifications in test_data
        """
        #np.argmax(y)
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        #pdb.set_trace()
        return sum(int(x == y) for (x, y) in test_results)


    def crossed(self, input):
        """ check if input picture is crossed
        """
        return (np.argmax(self.feedforward(input)))


#####
# General functions
def sigmoid(z):
    """ Sigmoid as activation function
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """ Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))

