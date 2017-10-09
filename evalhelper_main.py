#!/usr/bin/env python3

"""Evaluation helper - main program."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import glob
import pdb
import sys
import argparse

# Third-party libraries
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Own scripts
import evalhelper_func as eval
import evalhelper_dataloader as dl
import evalhelper_nn as nn
import evalhelper_statistics as st

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # get path to boegen files
    parser = argparse.ArgumentParser(description="Get file path.")
    parser.add_argument("-i", "--input", type=str)
    parser_args = parser.parse_args()
    path_to_boegen = parser_args.input
    if path_to_boegen is None:
        print("Error: Correct use of the script see below.")
        print("python3 evalhelper_main.py -i <PathToBoegenFiles>")
        sys.exit(0)

    # DO NOT CHANGE THE reference file: "boegen/Bogen3.jpg"
    reference_file = "boegen/Bogen3.jpg"

    # iter
    filecounter = 1

    # 1) IMAGE PART
    """
    #for current_file in glob.glob("boegen/*.jpg"):
    path_to_iterate = path_to_boegen + "/*.jpg"
    for current_file in glob.glob(path_to_iterate):

        # single / specific evaluations
        #if current_file == reference_file:
        #    continue
        # only for tests on ONE bogen
        #if current_file != "boegen/Bogen5.jpg":
        #    continue

        # 1.1) Reference points, transformation matrix and new positions
        # get reference points of the reference file
        refPoints = eval.findReferenceMasks("boegen/Bogen3.jpg")
        # ref points for bogen 3
        #refPoints = [[67, 689], [1125, 382], [2308, 374], [64, 2743], [2305, 2737]]

        # reference points of the current file
        newPoints = eval.findReferenceMasks(current_file)
        # new points for bogen 5
        #newPoints = [[70, 688], [1129, 380], [2315, 371], [68, 2737], [2312, 2731]]

        # get transformation matrix
        tmatrix = eval.getTransformationMatrix(refPoints=refPoints, newPoints=newPoints)
        # transformation matrix based on boegen 3 and 5
        #tmatrix = np.array([1.00239438e+00, 1.11612191e-03, -7.29325336e-04, 9.98115568e-01, 4.04140603e-04, -2.24452910e-03])

        # get new positions
        newpos = eval.getTransformedPositions(tmatrix, current_file)

        # 1.2) Find boxes
        # extract boxes as separate pictures from current bogen
        eval.extractBoxes(newpos, current_file)

        # show progress
        print(current_file)
        print("Progress: ", round(filecounter/28.0*100, 2))

        # iteration
        filecounter = filecounter + 1"""

    # 2) NN PART
    # 2.1) train NN
    # load training data
    # data needs to be in same folder as script
    trainingdata, testdata = dl.loaddata()

    # actual training of the NN
    evalnn = nn.Network([1600, 2])
    evalnn.SGD(trainingdata, testdata)

    # 2.2) Evaluation of boegen
    # manual check of one question
    #test_box = np.reshape(np.array(Image.open("boxes/Bogen5/box0.png").convert('L').getdata()) / 255, (1600,1))
    #print(evalnn.crossed(test_box))

    # storage of summary
    summarystats = np.zeros(shape=(28,14), dtype=int)
    # current path
    path = "boxes/Bogen"
    # loop through all boegen
    for bogencounter in range(28):
        # load bogen to be evaluated
        bogendata = dl.loadbogen(path + str(bogencounter+1))
        # evaluate answers
        evaluated = np.zeros(shape=(14,5,1), dtype=float)
        summary = np.zeros(shape=(14,), dtype=int)
        for j in range(14):
            for i in range(5):
                evaluated[j][i] = evalnn.crossed(np.reshape(bogendata[j][i], (1600,1)))
            # summary
            summary[j] = np.argmax(evaluated[j])
        # save in global summary
        summarystats[bogencounter] = summary

    # 2.3) Summary statistics
    # evaluate and print summary
    st.printStats(summarystats)

