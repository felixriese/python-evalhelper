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

    current_file = "boegen/Bogen5.jpg"

    # get reference points of the reference file
    # refPoints = eval.findReferenceMasks("boegen/Bogen3.jpg")
    # for Bogen3 (reference!):
    refPoints = [[67, 689], [1125, 382], [2308, 374], [64, 2743], [2305, 2737]]
    # print(refPoints)

    # TODO implement a for-loop for all the files

    # example: reference points of another file
    newPoints = eval.findReferenceMasks(current_file)
    # print(newPoints)

    # get transformation matrix
    tmatrix = eval.getTransformationMatrix(refPoints=refPoints,
                                           newPoints=newPoints)

    # get new positions
    newpos = eval.getTransformedPositions(tmatrix, current_file)

    # TODO now use the new position to read out the boxes
