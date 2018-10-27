#!/usr/bin/env python3

"""Evaluation helper part 3 - evaluate neural network."""

import pickle
import numpy as np
import glob

# Own scripts
import evalhelper_dataloader as dl
import evalhelper_statistics as st


if __name__ == "__main__":

    # load trained neural network
    evalnn = pickle.load(open("nn_object.p", "rb"))

    # manual check of one question
    # test_box = np.reshape(np.array(
    #     Image.open("boxes/Bogen5/box0.png").convert('L').getdata()) / 255,
    #     (1600,1))
    # print(evalnn.crossed(test_box))

    # current path of box pictures
    path = "boxes/Bogen"

    nfiles = len(glob.glob(path + "*"))
    nquestions = 14
    nanswers = 5

    # global summary -> 28 boegen, 14 questions each
    summarystats = np.zeros(shape=(nfiles, nquestions), dtype=int)

    # loop through all boegen
    for bogen in range(nfiles):

        # load bogen to be evaluated
        bogendata = dl.loadbogen(path + str(bogen+1))

        # evaluate each of the 5 boxes, crossed or not
        evaluated = np.zeros(shape=(nquestions, nanswers, 1), dtype=float)

        # for each of the 14 questions, indicated which answer/box is crossed
        # -> number between 0-4
        summary = np.zeros(shape=(nquestions,), dtype=int)

        for j in range(nquestions):
            for i in range(nanswers):

                # see if single box is crossed or not
                evaluated[j][i] = evalnn.crossed(np.reshape(bogendata[j][i],
                                                            (1600, 1)))
            # summary -> which of the boxes to one question is crossed
            summary[j] = np.argmax(evaluated[j])

        # save in global summary -> each row contains all answers of one bogen
        summarystats[bogen] = summary

    # evaluate and print summary
    st.printStats(summarystats)
