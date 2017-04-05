# importing needed libraries and functions
import os
from skimage import io

# importing project files
import preparation
import rectification
import runParameters

showProcess = runParameters.showProcess

for i in range(runParameters.start, runParameters.end + 1):
	runParameters.stepNumber = 0

	filename = os.path.join("../samples/"+str(i)+'.png')

	if showProcess:
		print(filename)

	preparation.setImageNumber(i);
	B = preparation.cleanAndBin(filename)

	rectification.setImageNumber(i);
	R = rectification.rectificate(B)

	resFile = os.path.join("results/"+str(i)+'.png')

	io.imsave(resFile, R*255)



