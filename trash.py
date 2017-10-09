#trash.py

# # scikit test
# mlp = MLPClassifier(hidden_layer_sizes=(10), batch_size=50,
#                     activation="logistic", solver="sgd",
#                     learning_rate_init=0.1, max_iter=15)
# mlp.fit([np.reshape(x[0], (1600,)) for x in trainingdata],
#         [np.reshape(x[1], (2,)) for x in trainingdata])
# y_test = mlp.predict([np.reshape(x[0], (1600,)) for x in trainingdata])
# pdb.set_trace()
# print(classification_report(np.array([np.reshape(x[1], (2,)) for x in trainingdata], dtype=int), y_test))
# # score = mlp.score([np.reshape(x[0], (1600,)) for x in testdata],
# #           [int(x[1][1]) for x in testdata])
# # print(score)
# sys.exit(0)
# # pdb.set_trace()


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