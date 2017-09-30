"""Evaluation helper - dataloader for NN."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import glob

# Third-party libraries
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def loaddata():
	# number of data
	nrd = 5573 + 21867
	# storage of data, nrd rows and 1600 columns (pixels)
	storage = np.ndarray(shape=(nrd,1600))

	# iterations
	i = 0

	# crossed ones (5573)
	for current_cross in glob.glob("crosses/work_type_crossed/*.png"):
		# convert greyscale and get value for each pixel
		cur_px = np.array(Image.open(current_cross).convert('L').getdata()) / 255
		# save in storage
		storage[i,] = cur_px
		# increase iteration
		i = i + 1

	# empty ones (21867)
	for current_empty in glob.glob("crosses/work_type_empty/*.png"):
		# convert greyscale and get value for each pixel
		cur_px = np.array(Image.open(current_cross).convert('L').getdata()) / 255
		# save in storage
		storage[i,] = cur_px
		# increase iteration
		i = i + 1

	# crossed vs empty
	yvalues = np.ndarray(shape=(nrd,1))
	yvalues[:,] = 0
	yvalues[0:5573,] = 1

	# return as tuple
	return (storage, yvalues)







