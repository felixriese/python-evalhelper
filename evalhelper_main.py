#!/usr/bin/env python3

"""Evaluation helper - main program."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import glob
import evalhelper_func as eval

# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # DO NOT CHANGE THE reference file: "boegen/Bogen3.jpg"
    reference_file = "boegen/Bogen3.jpg"

    for current_file in glob.glob("boegen/*.jpg"):

        if current_file == reference_file:
            continue

        # only for tests on ONE bogen
        if current_file != "boegen/Bogen5.jpg":
            continue

        # get reference points of the reference file
        # refPoints = eval.findReferenceMasks("boegen/Bogen3.jpg")
        refPoints = [[67, 689], [1125, 382], [2308, 374],
                     [64, 2743], [2305, 2737]]

        # reference points of the current file
        newPoints = eval.findReferenceMasks(current_file)

        # get transformation matrix
        tmatrix = eval.getTransformationMatrix(refPoints=refPoints,
                                               newPoints=newPoints)

        # get new positions
        newpos = eval.getTransformedPositions(tmatrix, current_file)

        # TODO now use the new position "newpos" to read out the boxes
