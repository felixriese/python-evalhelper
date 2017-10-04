#evalhelper_statistics.py

"""Evaluation helper - functions."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
# tbd

# Third-party libraries
import numpy as np

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------

def percentages(summary):
	""" percentage of answers
	"""
	# storage
	stats = np.zeros(shape=(14,5), dtype=float)
	# loop
	for qcounter in range(14):
		for acounter in range(5):
			stats[qcounter][acounter] = round(np.count_nonzero(summary[:,qcounter] == acounter) / float(28) * 100, 2)
	# return percentages
	return stats

