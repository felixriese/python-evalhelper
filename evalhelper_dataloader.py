"""Evaluation helper - dataloader for NN."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import glob
import random
import pdb

# Third-party libraries
import numpy as np
# print the entire array independent of its size
np.set_printoptions(threshold=np.nan)
from PIL import Image

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def loaddata():
    """ Load training and test data.

    Parameters
    ----------
    None : None

    Returns
    -------
    (training_data, test_data) : list of tuples
        tuple pairs with ndarrays for in and output
    """
    # number of data -> 27449
    nrd = 5573 + 21867
    # split : training vs test
    trdata_nrd = int(round(nrd*0.75))
    # trdata_nrd = nrd - 10001

    # storage of data, nrd rows and 1600 columns (pixels)
    storage = np.zeros(shape=(nrd, 1600), dtype=float)

    # list of pathnames for all pictures
    crossed_files = glob.glob("crosses/work_type_crossed/*.png")
    empty_files = glob.glob("crosses/work_type_empty/*.png")
    files = crossed_files + empty_files

    # loop through all
    for counter in range(len(files)):
        # convert greyscale & get value (0-255) for each pixel (scaled to 0-1)
        cur_px = np.array(Image.open(files[counter]).convert('L').getdata()) / float(255)
        # save in storage
        storage[counter] = cur_px

    # crossed vs empty -> desired output values
    yvalues = np.zeros(shape=(nrd, 2), dtype=float)
    for j in range(nrd):
        if j <= 5573:
            # crossed is (0,1) vector
            yvalues[j] = np.array([0, 1])
        else:
            # uncrossed is (1,0) vector
            yvalues[j] = np.array([1, 0])

    # return as list of tuples (x,y) whereas x = 1600x1 np.array, y = 2x1 np.array
    dataset = [(np.reshape(s, (1600, 1)), np.reshape(y, (2, 1)))
               for s, y in zip(storage, yvalues)]
    # shuffle before split with fixed seed
    random.seed(1)
    random.shuffle(dataset)
    # split into training and test data
    training_data = dataset[:trdata_nrd]
    test_data = dataset[trdata_nrd:]
    # return
    return (training_data, test_data)


def loadbogen(path):
    """ Load a boegen that is to be evaluated.

    Parameters
    ----------
    path : str
        path to bogen

    Returns
    -------
    storage: ndarray
        contains all 14 question with 5 boxes each

    """
    # storage of data, 14 "rows" with 5x1600 arrays in each (answers x pixels)
    storage = np.zeros(shape=(14,5,1600), dtype=float)

    # current path
    current_bogen = path + "/box"

    # iterations
    i = 0
    j = 0

    # loop through all 70 boxes
    for counter in range(70):
        # current_box
        current_box = current_bogen + str(counter) + ".png"
        # convert greyscale and get value (0-255) for each pixel (scaled to 0-1)
        cur_px = np.array(Image.open(current_box).convert('L').getdata()) / float(255)
        # save in storage
        storage[i][j] = cur_px
        # increase iteration
        if (j+1) % 5 == 0:
            i = i + 1
            j = 0
        else:
            j = j + 1

    # return data
    return storage
