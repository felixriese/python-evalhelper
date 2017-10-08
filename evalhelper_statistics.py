#!/usr/bin/env python3

"""Evaluation helper - functions."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def printStats(summary):

    # storage
    stats = np.zeros(shape=(14,5), dtype=float)
    # loop
    for qcounter in range(14):
        for acounter in range(5):
            stats[qcounter][acounter] = round(np.count_nonzero(summary[:,qcounter] == acounter) / float(28) * 100, 2)

    print("###\nBoegen - Evaluation\n###")
    print("Q\t1\t2\t3\t4\t5\t|  Mean")
    for i, question in enumerate(stats):
        # print(question)
        print("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t|  %.2f" %
              (i+1, question[0], question[1], question[2], question[3],
               question[4], getAverage(question)))


def getAverage(question):
    av = 0
    for i, a in enumerate(question):
        av += (i+1) * a / 100
    return av
