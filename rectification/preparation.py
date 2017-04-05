from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
import os

import runParameters

showProcess = runParameters.showProcess
writeProcessImages = runParameters.writeProcessImages

imageNumber = 0

def setImageNumber(f):
	global imageNumber
	imageNumber = f

def cleanAndBin(filename):
	im = io.imread(filename)

	# remove borders to mask
	im = im[4:-4,4:-4]

	im = rgb2gray(im)

	val = filters.threshold_otsu(im)
	mask = im < val

	if showProcess:
		#io.imshow(mask)
		viewer = ImageViewer(mask)
		viewer.show()

	if writeProcessImages:
		folder = str(runParameters.stepNumber) + "_binarization";
		if not os.path.exists(folder):
			os.makedirs(folder)
		io.imsave(folder + "/" + str(imageNumber) + ".png", mask*255)

	runParameters.stepNumber += 1

	return mask
