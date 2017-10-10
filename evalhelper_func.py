#!/usr/bin/env python3

"""Evaluation helper - functions."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import csv
import os
import pdb

# Third-party libraries
from PIL import Image, ImageDraw
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def getReference(boegen_folder, doDraw=False):
    """Get reference points for the chosen file.

    Parameters
    ----------
    boegen_folder : str
        Path to folder of boegen
    doDraw : bool, optional
        If true, the reverence evaluation file is drawn
        
    Returns
    -------
    refpos : dict
        Dictionary of all reference positions

    """

    path = boegen_folder + "/Bogen3.jpg"
    # positions of actual boxes on reference bogen
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
    #im.show()


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
    #print(posdict)
    return posdict


def getMSEBetweenTwoImages(im1, im2, x_max=36, y_max=36):
    """Calculate mean squared error (MSE) between two images pixel-wise.

    Parameters
    ----------
    im1, im2 : images
        Images to be compared with
    x_max, y_max : int, optional
        Maximum values of x and y

    Returns
    -------
    float
        Mean squared error

    """

    mse = 0
    for i in range(y_max):
        for j in range(x_max):
            # compare pixel wise, ie "value" of each pixel
            mse += (im1[i, j] - im2[i, j])**2
    return mse/(y_max*x_max)


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
    mse_min = 999999.
    # scan through predefined area of bogen picture
    for x in range(30, 110):
        for y in range(650, 730):
            # extract rectangular on which corner is searched
            px = im.crop([x, y, x+52, y+52]).load()
            mse_curr = getMSEBetweenTwoImages(mask_ul_px, px, 52, 52)
            if mse_curr < mse_min:
                mse_min = mse_curr
                # LL pixel position
                pos_ul = [x, y]
    # print(mse_min, pos_ul)

    # mask upper middle
    mask_um = Image.open("masks/ul.png")
    mask_um_px = mask_um.load()
    pos_um = [0, 0]
    mse_min = 999999.
    for x in range(1070, 1150):
        for y in range(340, 430):
            px = im.crop([x, y, x+52, y+52]).load()
            mse_curr = getMSEBetweenTwoImages(mask_um_px, px, 52, 52)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_um = [x, y]
    # print(mse_min, pos_um)

    # mask upper right
    mask_ur = Image.open("masks/ur.png")
    mask_ur_px = mask_ur.load()
    pos_ur = [0, 0]
    mse_min = 999999
    for x in range(2300, 2400):
        for y in range(350, 430):
            px = im.crop([x, y, x+54, y+59]).load()
            mse_curr = getMSEBetweenTwoImages(mask_ur_px, px, 54, 54)
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
        for y in range(2720, 2791):
            px = im.crop([x, y, x+52, y+36]).load()
            mse_curr = getMSEBetweenTwoImages(mask_ll_px, px, 36, 36)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_ll = [x, y]
    # print(mse_min, pos_ll)

    # mask lower right
    mask_lr = Image.open("masks/lr.png")
    mask_lr_px = mask_lr.load()
    pos_lr = [0, 0]
    mse_min = 999999
    for x in range(2300, 2400):
        for y in range(2710, 2789):
            px = im.crop([x, y, x+51, y+54]).load()
            mse_curr = getMSEBetweenTwoImages(mask_lr_px, px, 51, 51)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_lr = [x, y]
    # print(mse_min, pos_lr)

    return pos_ul, pos_um, pos_ur, pos_ll, pos_lr


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
        A = x[:4].reshape(2, 2)
        b = x[4:]
        return sum([(norm(np.dot(A, refPoints[i]) + b - newPoints[i]))**2
                    for i in range(len(refPoints))])

    x0 = [1., 0., 0., 1., 0., 0.]
    res = minimize(errFunc, x0, method="nelder-mead",
                   options={'maxiter': 10000000})   # 'xatol': 1e-12,
    #print("tmatrix = ", res.x)
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


def getTransformedPositions(tmatrix, path, boegen_folder):
    """Get new box positions of file.

    Parameters
    ----------
    tmatrix : list of float
        Matrix A and vector b of the linear transformation
    path : str
        Path to file
    boegen_folder : str
        Path to folder of boegen

    Returns
    -------
    refpos :  : dict
        New dictionary of all reference positions

    """

    refpos = getReference(boegen_folder, doDraw=True)

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

    #print(refpos)
    drawFile(refpos=refpos, path=path)
    return refpos


def extractBoxes(refpos, path):
    """Extract boxes from current file.

    Parameters
    ----------
    refpos : dict
        Dictionary of all reference positions
    path : str
        Path to file

    Returns
    -------
    images : png
        Boxes as images

    """

    # load image
    im = Image.open(path)

    # loop for rectangles
    for i in range(len(refpos["x0"])-4):
        # get corner points
        corners = [refpos["x0"][i]-10, refpos["y0"][i]-10,
                   refpos["x1"][i]+10, refpos["y1"][i]+10]
        # cut box out of image
        box = im.crop(corners)
        # resize to 40x40 image
        box = box.resize((40,40))
        # iterative folder
        # necessary as glob.glob has a kind of random sorting
        folder = path[path.find("/Bogen")+1:path.find(".")]
        # path and folder
        directory = "boxes/" + folder
        # check if folder exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        # iterative name
        name = "box" + str(i)
        # save as image
        box.save(directory + "/" + name + ".png")



