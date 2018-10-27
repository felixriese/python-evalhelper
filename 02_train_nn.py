#!/usr/bin/env python3

"""Evaluation helper part 2 - train neural network."""

import pickle

# Own scripts
import evalhelper_dataloader as dl
import evalhelper_nn as nn


if __name__ == "__main__":

    # load training data
    # data needs to be in same folder as script
    trainingdata, testdata = dl.loaddata()

    # actual training of the NN
    evalnn = nn.Network([1600, 2])
    evalnn.SGD(trainingdata, testdata)

    # save nn via pickle
    pickle.dump(evalnn, open("nn_object.p", "wb"))
    print("Saved neural network to file.")
