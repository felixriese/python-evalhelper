"""Evaluation helper - dataloader for NN."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import glob
import random

# Third-party libraries
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def loaddata():
	# number of data
	nrd = 5573 + 21867
	# split : training vs test
	trdata_nrd = round(nrd*0.75)

	# storage of data, nrd rows and 1600 columns (pixels)
	storage = np.zeros(shape=(nrd,1600))

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
	yvalues = np.zeros(shape=(nrd,2))
	for j in range(nrd):
		if j <= 5573:
			yvalues[j,] = np.array([0,1])
		else:
			yvalues[j,] = np.array([1,0])

	# return as list of tuples (x,y) whereas x = 1600x1 np.array, y = 2x1 np.array
	dataset = [(np.reshape(s,(1600,1)),np.reshape(y,(2,1))) for s,y in zip(storage, yvalues)]
	# shuffle before split
	random.shuffle(dataset)
	# split into training and test data
	training_data = dataset[0:trdata_nrd].copy()
	test_data = dataset[trdata_nrd+1:].copy()
	# return
	return (training_data, test_data)







