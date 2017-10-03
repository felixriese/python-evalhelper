#!/usr/bin/env python3

"""Evaluation helper - main program."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import glob

# Third-party libraries
import numpy as np

# Own scripts
import evalhelper_func as eval
import evalhelper_dataloader as dl
import evalhelper_nn as nn

# ---------------------------------------------------------------------------
# Main routine
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

        # 1) Reference points, transformation matrix and new positions
        # get reference points of the reference file
        # refPoints = eval.findReferenceMasks("boegen/Bogen3.jpg")
        refPoints = [[67, 689], [1125, 382], [2308, 374], [64, 2743], [2305, 2737]]

        # reference points of the current file
        #newPoints = eval.findReferenceMasks(current_file)
        newPoints = [[70, 688], [1129, 380], [2315, 371], [68, 2737], [2312, 2731]]

        # get transformation matrix
        #tmatrix = eval.getTransformationMatrix(refPoints=refPoints, newPoints=newPoints)
        tmatrix = np.array([1.00239438e+00, 1.11612191e-03, -7.29325336e-04, 9.98115568e-01, 4.04140603e-04, -2.24452910e-03])

        # get new positions
        newpos = eval.getTransformedPositions(tmatrix, current_file)

        # 2) Find boxes
        # extract boxes as separate pictures from current bogen
        eval.extractBoxes(newpos, current_file)

    # 3) NN PART
    # load data
    trainingdata, test_data = dl.loaddata()
    
    # train NN
    evalnn = nn.Network([1600, 2])
    evalnn.SGD(trainingdata, test_data)





        