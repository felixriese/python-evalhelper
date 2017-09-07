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
    # eval.getReference()
    # # eval.getRefPositions("reference_positions.csv")

    print(eval.findReferenceMasks("boegen/Bogen3.jpg"))
