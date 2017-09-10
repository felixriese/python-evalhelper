#!/usr/bin/env python3

"""Evaluation helper - main program."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import evalhelper_func as eval

# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # DO NOT CHANGE THE reference file: "boegen/Bogen3.jpg"

    # get reference points of the reference file
    # refPoints = eval.findReferenceMasks("boegen/Bogen3.jpg")
    refPoints = [[67, 689], [2312, 374], [64, 2743], [2312, 2741]]
    # print(refPoints)

    # TODO implement a for-loop for all the files

    # example: reference points of another file
    # newPoints = eval.findReferenceMasks("boegen/Bogen1.jpg")
    newPoints = [[68, 690], [2303, 372], [67, 2722], [2318, 2745]]
    # print(newPoints)

    # get transformation matrix
    tmatrix = eval.getTransformationMatrix(refPoints=refPoints,
                                           newPoints=newPoints)

    # get new positions
    newpos = eval.getTransformedPositions(tmatrix, "boegen/Bogen1.jpg")

    # TODO now use the new position to read out the boxes
