#!/usr/bin/env python3

"""Evaluation helper part 1 - process files."""

# Standard libraries
import glob
import sys
import argparse

# Own scripts
import bogen


if __name__ == "__main__":

    # get path to boegen files
    parser = argparse.ArgumentParser(description="Get file path.")
    parser.add_argument("-i", "--input", type=str)
    parser_args = parser.parse_args()
    path_to_boegen = parser_args.input
    if path_to_boegen is None:
        print("Error: Correct use of the script see below.")
        print("    python process_files.py -i <PathToBoegenFiles>")
        sys.exit(0)

    # DO NOT CHANGE THE reference file: "boegen/Bogen3.jpg"
    reference_file = path_to_boegen + "/Bogen3.jpg"

    # reference points of the reference file (bogen3)
    # -> position of masks/corners
    referenceBogen = bogen.Bogen(curr_file=reference_file,
                                 ref_points=None,
                                 ref_boxes=None,
                                 isReference=True)
    refPoints = referenceBogen.ReferencePoints
    refBoxPositions = referenceBogen.ReferenceBoxes

    path_to_iterate = path_to_boegen + "/*.jpg"
    nfiles = len(glob.glob(path_to_iterate))
    for i, current_file in enumerate(glob.glob(path_to_iterate)):

        curr_bogen = bogen.Bogen(curr_file=current_file, ref_points=refPoints,
                                 ref_boxes=refBoxPositions, isReference=False)

        # show progress
        print("Progress: ", round((i + 1) / nfiles * 100, 2),
              "(", current_file, ")")
