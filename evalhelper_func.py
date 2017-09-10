#!/usr/bin/env python3

"""Evaluation helper - functions."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from PIL import Image, ImageDraw
import csv
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def getReference(doDraw=False):
    """Get reference points for the chosen file.

    Parameters
    ----------
    doDraw : bool, optional
        If true, the reverence evaluation file is drawn

    Returns
    -------
    refpos : dict
        Dictionary of all reference positions

    """
    path = "boegen/Bogen3.jpg"
    refpos = getRefPositions("reference_positions.csv")

    if doDraw:
        drawFile(refpos, path)

    return refpos


def drawFile(refpos, path):
    """Draw file with all rectangles.

    Parameters
    ----------
    refpos : dict
        Dictionary of all reference positions
    path : str
        Path to file

    """
    im = Image.open(path)
    draw = ImageDraw.Draw(im)
    for i in range(len(refpos["x0"])):
        box = [refpos["x0"][i], refpos["y0"][i],
               refpos["x1"][i], refpos["y1"][i]]

        draw.rectangle(box, fill="Black")
    im.show()


def getRefPositions(path):
    """Get reference positions from CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file containing the positions of the reference file.

    Returns
    -------
    posdict : dict
        Dictionary containing the positions of the reference file.

    """
    posdict = {}

    with open(path) as posfile:
        reader = csv.DictReader(posfile, delimiter="\t",
                                skipinitialspace=True)
        for i, row in enumerate(reader):
            for c, v in row.items():
                if c == "question" or c == "box":
                    posdict.setdefault(c, []).append(v)
                else:
                    posdict.setdefault(c, []).append(int(v))
    # print(posdict)
    return posdict


def getMSEBetweenTwoImages(im1, im2):
    """Calculate mean squared error (MSE) between two images pixel-wise.

    Parameters
    ----------
    im1, im2 : images
        Images to be compared with

    Returns
    -------
    float
        Mean squared error

    """
    mse = 0
    for i in range(36):
        for j in range(36):
            mse += (im1[i, j] - im2[i, j])**2
    return mse/(36*36)


def findReferenceMasks(image):
    """Find positions of the corners (see masks).

    The images are formated as 2479x2829.

    Parameters
    ----------
    image : str
        Path to image

    Returns
    -------
    lists of int
        Positions [x,y] of the reference points (corners).

    """
    im = Image.open(image)

    # mask upper left
    mask_ul = Image.open("masks/ul.png")
    mask_ul_px = mask_ul.load()
    pos_ul = [0, 0]
    mse_min = 999999
    for x in range(30, 110):
        for y in range(650, 730):
            px = im.crop([x, y, x+40, y+40]).load()
            mse_curr = getMSEBetweenTwoImages(mask_ul_px, px)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_ul = [x, y]
    # print(mse_min, pos_ul)

    # mask upper right
    mask_ur = Image.open("masks/ur.png")
    mask_ur_px = mask_ur.load()
    pos_ur = [0, 0]
    mse_min = 999999
    for x in range(2300, 2380):
        for y in range(350, 430):
            px = im.crop([x, y, x+40, y+40]).load()
            mse_curr = getMSEBetweenTwoImages(mask_ur_px, px)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_ur = [x, y]
    # print(mse_min, pos_ur)

    # mask lower left
    mask_ll = Image.open("masks/ll.png")
    mask_ll_px = mask_ll.load()
    pos_ll = [0, 0]
    mse_min = 999999
    for x in range(30, 110):
        for y in range(2720, 2789):
            px = im.crop([x, y, x+40, y+40]).load()
            mse_curr = getMSEBetweenTwoImages(mask_ll_px, px)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_ll = [x, y]
    # print(mse_min, pos_ll)

    # mask upper right
    mask_lr = Image.open("masks/lr.png")
    mask_lr_px = mask_lr.load()
    pos_lr = [0, 0]
    mse_min = 999999
    for x in range(2300, 2380):
        for y in range(2710, 2789):
            px = im.crop([x, y, x+40, y+40]).load()
            mse_curr = getMSEBetweenTwoImages(mask_lr_px, px)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_lr = [x, y]
    # print(mse_min, pos_lr)

    return pos_ul, pos_ur, pos_ll, pos_lr


def getTransformationMatrix(refPoints, newPoints):
    """Get transformation matrix between reference and new file.

    Parameters
    ----------
    refPoints : list of lists
        Reference points of reference file
    newPoints : list of lists
        Reference points of current file

    Returns
    -------
    list of lists
        Matrix A and vector b of the linear transformation

    """
    def errFunc(x):
        A = x[:len(refPoints)].reshape(2, 2)
        b = x[len(refPoints):]
        return sum([(norm(np.dot(A, refPoints[i]) + b - newPoints[i]))**2
                    for i in range(len(refPoints))])

    x0 = [1., 1., 1., 1., 0., 0.]
    res = minimize(errFunc, x0, method="nelder-mead", options={'xtol': 1e-8})
    print(res.x)
    return res.x


def transformPoint(tmatrix, point):
    """Transform point with transformation matrix.

    Parameters
    ----------
    tmatrix : list of float
        Matrix A and vector b of the linear transformation
    point : list of int
        2D point to be transformed

    Returns
    -------
    newpoint : list of float
        Transformed 2D point

    """
    A = tmatrix[:4].reshape(2, 2)
    b = tmatrix[4:]
    newpoint = np.dot(A, point) + b
    return newpoint


def getTransformedPositions(tmatrix, path):
    """Get new box positions of file.

    Parameters
    ----------
    tmatrix : list of float
        Matrix A and vector b of the linear transformation
    path : str
        Path to file

    Returns
    -------
    refpos :  : dict
        New dictionary of all reference positions

    """
    refpos = getReference(doDraw=True)

    # transform point for point
    for i in range(len(refpos["x0"])):
        pointUL = [refpos["x0"][i], refpos["y0"][i]]
        newPointUL = transformPoint(tmatrix, pointUL)

        pointLR = [refpos["x1"][i], refpos["y1"][i]]
        newPointLR = transformPoint(tmatrix, pointLR)

        refpos["x0"][i] = newPointUL[0]
        refpos["y0"][i] = newPointUL[1]
        refpos["x1"][i] = newPointLR[0]
        refpos["y1"][i] = newPointLR[1]

    # print(refpos)
    drawFile(refpos=refpos, path=path)
    return refpos
