#!/usr/bin/env python3

"""Evaluation helper - functions."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import pdb

# Third-party libraries
import numpy as np

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def printStats(summary):
    """ Print statistics of evaluated boegen.

    Parameters
    ----------
    summary : ndarray
        evaluation results

    Returns
    -------
    None : None
        just printing of results

    """
    # storage -> 14x5 as 14 questions and 5 answers each
    stats = np.zeros(shape=(14, 5), dtype=float)
    # loop through all
    for qcounter in range(14):
        for acounter in range(5):
            # count non_zeros equal to 0-4 in each column (ie all rows)
            stats[qcounter][acounter] = round(np.count_nonzero(
                summary[:, qcounter] == acounter) / float(28) * 100, 2)
    # print stats
    print("\n\n###\nBoegen - Evaluation\n###")
    print("Q\t1\t2\t3\t4\t5\t|  Mean")
    for i, question in enumerate(stats):
        # print(question)
        print("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t|  %.2f" %
              (i+1, question[0], question[1], question[2], question[3],
               question[4], getAverage(question)))


def getAverage(question):
    """ Calculate average for a question.

    Parameters
    ----------
    question : ndarray
        answers that are to be averaged

    Returns
    -------
    av : float
        average result

    """
    av = 0
    for i, a in enumerate(question):
        av += (i+1) * a / 100
    return av
