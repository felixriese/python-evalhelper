#!/usr/bin/env python3

"""Evaluation helper - functions."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from PIL import Image, ImageDraw, ImageColor
import csv

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def getReference():
    path = "boegen/Bogen3.jpg"
    refpos = getRefPositions("reference_positions.csv")

    im = Image.open(path)
    # px = im.load()
    draw = ImageDraw.Draw(im)

    for i in range(len(refpos["x0"])):
        box = [refpos["x0"][i], refpos["y0"][i],
               refpos["x1"][i], refpos["y1"][i]]
        # rectangle: [x0, y0, x1, y1]
        # box = (1000, 1000, 1200, 1200)
        # region = im.crop(box)
        # region.show()

        draw.rectangle(box, fill="Black")
    im.show()

    return refpos


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
    print(posdict)
    return posdict


def getMSEBetweenTwoImages(im1, im2):
    """Calculate mean squared error (MSE) between two images pixel-wise."""
    mse = 0
    for i in range(36):
        for j in range(36):
            mse += (im1[i, j] - im2[i, j])**2
    return mse/1600


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
    for x in range(30, 120):
        for y in range(650, 750):
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
    for x in range(2300, 2400):
        for y in range(350, 450):
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
    for x in range(30, 120):
        for y in range(2710, 2789):
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
    for x in range(2300, 2400):
        for y in range(2710, 2789):
            px = im.crop([x, y, x+40, y+40]).load()
            mse_curr = getMSEBetweenTwoImages(mask_lr_px, px)
            if mse_curr < mse_min:
                mse_min = mse_curr
                pos_lr = [x, y]
    # print(mse_min, pos_lr)

    return pos_ul, pos_ur, pos_ll, pos_lr
