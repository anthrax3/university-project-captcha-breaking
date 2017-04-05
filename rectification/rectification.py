from skimage import io
from skimage.viewer import ImageViewer
import numpy as np
from skimage.morphology import binary_closing
from skimage.morphology import binary_erosion
import os

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import runParameters

showProcess = runParameters.showProcess
writeProcessImages = runParameters.writeProcessImages

imageNumber = 0;

def setImageNumber(f):
	global imageNumber
	imageNumber = f

def close(B):
	neiDia = 20 # neighborhood square side

	# expanding image
	C = [0] * (2 * neiDia + len(B[0]))
	C = [C] * (2 * neiDia + len(B))
	C = np.array(C)
	C[neiDia:-neiDia, neiDia:-neiDia] = B[:, :]

	# creating a 2d array of ones
	ones = [1] * neiDia
	ones = [ones] * neiDia

	C = binary_closing(C, ones)

	# returning to original size
	C = C[neiDia:-neiDia, neiDia:-neiDia]

	if showProcess:
		viewer = ImageViewer(C)
		viewer.show()

	if writeProcessImages:
		folder = str(runParameters.stepNumber) + "_closing";
		if not os.path.exists(folder):
			os.makedirs(folder)
		io.imsave(folder + "/" + str(imageNumber) + ".png", C*255)
	
	runParameters.stepNumber += 1

	return C

def erode(C):
	eroDia = 10

	# creating a 2d array of ones
	ones = [1] * 1
	ones = [ones] * eroDia



	E = binary_erosion(C, ones)

	if showProcess:
		viewer = ImageViewer(E)
		viewer.show()

	if writeProcessImages:
		folder = str(runParameters.stepNumber) + "_eroding";
		if not os.path.exists(folder):
			os.makedirs(folder)
		io.imsave(folder + "/" + str(imageNumber) + ".png", E*255)
	
	runParameters.stepNumber += 1


	return E


def getPoints(E):
	vals = []
	for x in range(len(E)):
		for y in range(len(E[x])):
			if E[x, y]:
				vals.append([x, y])

	points = np.array(vals, dtype='d')

	if showProcess:
		print( "# of points: " + str(len(points)))
		print( "max row: " + str(np.max(points[:,0])) )
		print( "min row: " + str(np.min(points[:,0])) )
		print( "max col: " + str(np.max(points[:,1])) ) 
		print( "min col: " + str(np.min(points[:,1])) )

	return points


def getTrainedModel(points):
	degree = 5 # degree of function to fit
	maxSampleSize = 250

	sampleSize = min(maxSampleSize, len(points))

	# getting train data
	rng = np.random.RandomState()
	rng.shuffle(points)
	y = points[:sampleSize,0]
	x = points[:sampleSize,1]

	# create matrix versions of these arrays
	X = x[:, np.newaxis]


	# ridge interpolation: 
	# http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
	model = make_pipeline(PolynomialFeatures(degree), Ridge())
	model.fit(X, y)
	
	if showProcess:
		# generate points used to plot
		x_plot = np.linspace(np.min(x) - 5, np.max(x) + 5, 100) # de 0 a 200 inclusive tomando 100 puntos

		X_plot = x_plot[:, np.newaxis]
		y_plot = model.predict(X_plot)

		plt.scatter(x, y, label="training points")
		plt.plot(x_plot, y_plot, label="degree %d" % degree)
		plt.legend(loc='lower left')
		plt.gca().invert_yaxis()
		plt.show()

	return model

def align(B, me, model):
	
	A = [0] * (len(B[0]))
	A = [A] * (len(B))
	A = np.array(A)

	offset = me[0]

	for i in range(len(B[0])):
		y = model.predict([[i]])
		for j in range(len(B)):
			pj = j + offset - y # projection
			pj = (int)(pj)
			if pj >= 0 and pj < len(B):
				A[pj, i] = B[j, i]

	if showProcess:
		viewer = ImageViewer(A.astype(bool))
		viewer.show()

	if writeProcessImages:
		folder = str(runParameters.stepNumber) + "_alignment";
		if not os.path.exists(folder):
			os.makedirs(folder)
		io.imsave(folder + "/" + str(imageNumber) + ".png", A*255)
	
	runParameters.stepNumber += 1

	return A

def rectificate(B):
	C = close(B)

	E = erode(C)

	points = getPoints(E)

	model = getTrainedModel(points)

	A = align(B, np.mean(points, axis = 0), model)

	return A