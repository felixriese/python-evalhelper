"""Evaluation helper - dataloader for NN."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import glob
import random
import pdb

# Third-party libraries
import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.nan)

# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def loaddata():
	""" load training and test data
	"""

	# number of data
	nrd = 5573 + 21867
	# split : training vs test
	#trdata_nrd = int(round(nrd*0.75))
	trdata_nrd = nrd - 10001

	# storage of data, nrd rows and 1600 columns (pixels)
	storage = np.zeros(shape=(nrd,1600), dtype=float)

	# iterations
	#i = 0

	# iterative filex
	crossed_files = glob.glob("crosses/work_type_crossed/*.png")
	empty_files = glob.glob("crosses/work_type_empty/*.png")
	files = crossed_files + empty_files

	# loop through all
	for counter in range(len(files)):
		# convert greyscale and get value for each pixel
		cur_px = np.array(Image.open(files[counter]).convert('L').getdata()) / float(255)
		# save in storage
		storage[counter] = cur_px

	"""# crossed ones (5573)
	for current_cross in glob.glob("crosses/work_type_crossed/*.png"):
		# convert greyscale and get value for each pixel
		cur_px = np.array(Image.open(current_cross).convert('L').getdata()) / float(255)
		# save in storage
		storage[i,] = cur_px
		# increase iteration
		i = i + 1

	# empty ones (21867)
	for current_empty in glob.glob("crosses/work_type_empty/*.png"):
		# convert greyscale and get value for each pixel
		cur_px = np.array(Image.open(current_cross).convert('L').getdata()) / float(255)
		# save in storage
		storage[i,] = cur_px
		# increase iteration
		i = i + 1"""

	# crossed vs empty
	yvalues = np.zeros(shape=(nrd,2), dtype=float)
	for j in range(nrd):
		if j <= 5573:
			yvalues[j] = np.array([0,1])
		else:
			yvalues[j] = np.array([1,0])

	# return as list of tuples (x,y) whereas x = 1600x1 np.array, y = 2x1 np.array
	dataset = [(np.reshape(s,(1600,1)), np.reshape(y,(2,1))) for s,y in zip(storage, yvalues)]
	# shuffle before split
	#training_inputs = [np.reshape(x, (1600, 1)) for x in storage]
	#training_results = [np.reshape(y, (2,1)) for y in yvalues]
	#dataset = zip(training_inputs, training_results)
	random.shuffle(dataset)
	# split into training and test data
	training_data = dataset[:trdata_nrd]
	test_data = dataset[trdata_nrd+1:]
	#pdb.set_trace()
	# return
	return (training_data, test_data)


def loadbogen(path):
	""" load a boegen that is to be evaluated
	"""
	# storage of data, 14 rows with 5x1600 arrays in each (answers x pixels)
	storage = np.zeros(shape=(14,5,1600), dtype=float)

	# current path
	current_bogen = path + "/box"

	# iterations
	i = 0
	j = 0

	for counter in range(70):
		# current_box
		current_box = current_bogen + str(counter) + ".png"
		# convert greyscale and get value for each pixel
		cur_px = np.array(Image.open(current_box).convert('L').getdata()) / float(255)
		# save in storage
		storage[i][j] = cur_px
		# increase iteration
		if (j+1)%5==0:
			i = i + 1
			j = 0
		else:
			j = j + 1

	# return data
	return storage









