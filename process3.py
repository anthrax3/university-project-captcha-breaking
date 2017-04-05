# This process maximizes the confidence given by the OCR

import skimage
from skimage import data
from skimage import io
from skimage import filters
import os
from skimage import viewer
from skimage.color import rgb2gray
from scipy.spatial import ConvexHull
import numpy as np
from skimage.transform import rotate

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

import matplotlib.pyplot as plt

import math

import tesseract

#showProcess = True
showProcess = False
NUMBER_OF_IMAGES = 2000

showf = False


api = tesseract.TessBaseAPI()
api.Init("/usr/share/tesseract-ocr","eng",tesseract.OEM_DEFAULT)
#api.SetPageSegMode(tesseract.PSM_AUTO)
api.SetPageSegMode(tesseract.PSM_SINGLE_WORD)
api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

f = open("out3.txt", "w")

def getDistance(points, angle, center):
	global showf
	A = center
	B = np.array([0,0], dtype='d') + [math.cos(angle), math.sin(angle)]
	mini = 0.0
	maxi = 0.0
	if showf:
		print ("B: " + str(B))
		print ("A: " + str(A))
	for i in range(len(points)):
		C = points[i]
		t = np.dot(C - A, B)
		if showf:
			print ("C: "+ str(C) )
			print ("t: " + str(t))
		mini = min(mini, t)
		maxi = max(maxi, t)
	showf = False;
	return (mini, maxi)

def getEntropy(points, angle, center, dist):
	global showProcess
	A = center
	B = np.array([0,0], dtype='d') + [math.cos(angle), math.sin(angle)]
	totalDistance = dist[1] - dist[0]
	diff = 10 # interval to get the discrete histogram
	histogram = [0 for x in range( (int)(totalDistance/diff) + 1)]
	n = len(points)
	for i in range(n):
		C = points[i]
		t = np.dot(C - A, B)
		t -= dist[0]
		histogram[ (int)(t/diff) ] += 1;
	
	if showProcess:
		print(str(histogram))

	entropy = 0.0
	for f in histogram:
		prob = (float)(f)/(float)(n)
		if prob == 0.0:
			continue
		entropy += prob*math.log(prob)

	entropy = -1.0 * entropy
	return entropy

def getConfidence(mask, angle, me):
	aDeg = math.degrees(angle)
	rotated = rotate(mask, 90.0 - aDeg, False, (me[1],me[0]))

	#rotated = rotate(mask, 90.0-adeg, False)
	io.imsave("temp3_0.png", rotated)
	
	#using the OCR
	gifTempFile = "temp3_1.gif"
	tifTempFile = "temp3_2.tif"
	im = Image.open('temp3_0.png')
	im.save(gifTempFile, "GIF")
	original = Image.open(gifTempFile)
	bg = original.resize(im.size, Image.NEAREST)
	bg.save(tifTempFile)

	pixImage=tesseract.pixRead(tifTempFile)
	api.SetImage(pixImage)
	outText=api.GetUTF8Text()
	conf=api.MeanTextConf()

	if showProcess:
		wait = input("PRESS ENTER TO CONTINUE.")

	if conf <= 10:
		conf = 10
	return conf
	
def ternary_search(start, end, mask, me, maxError):
	a = start
	b = end
	error = b - a
	
	if showProcess:
		print("")
		print("[ternary_search] start: "+str(math.degrees(start)) + " end: " + str( math.degrees(end) ))
	
	evalA = 0
	while error > maxError:
		an = (2*a + b)/3
		bn = (a + 2*b)/3
		
		#distA = getDistance(points[hull.vertices], an, me)
		evalA = getConfidence(mask, an, me)

		#distB = getDistance(points[hull.vertices], bn, me)
		evalB = getConfidence(mask, bn, me)
		
		if showProcess:
			print(str(math.degrees(an)) + " " + str(evalA))
		
		if evalA < evalB:
			a = an
		else:
			b = bn
			
		error = b - a
	return (a, evalA)


# main Process
for i in range(NUMBER_OF_IMAGES):
	filename = os.path.join("samples/"+str(i)+'.png')
	if showProcess:
		print(filename)
	image = io.imread(filename)
	origImage = image
	image = rgb2gray(image)
	
	val = filters.threshold_otsu(image)
	mask = image < val
	
	#remove borders to mask
	mask = mask[4:-4,4:-4]

	vals = []
	for x in range(len(mask)):
		for y in range(len(mask[x])):
			if mask[x][y]:
				vals.append([x, y])

	points = np.array(vals, dtype='d')
	hull = ConvexHull(points)
	
	if showProcess:
		plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
		plt.plot(points[:,0], points[:,1],'ro')
		plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
		plt.show()

	me = np.mean(points, axis=0)

	if showProcess:
		print("mean: " + str(me))

	# ternary search
	MAX_ERROR = 0.00001 #max allowed error, aprox. 0.005 degrees
	(a, evalA) = ternary_search(0, math.pi, mask, me, MAX_ERROR)
	
	adeg = math.degrees(a)
	if showProcess:
		print("angle found: " + str(a))
		print("angle found(degrees): " + str(adeg))
		print("max confidence found: " + str(evalA))	

	rotated = rotate(mask, 90.0 - adeg, False, (me[1], me[0]))

	#rotated = rotate(mask, 90.0-adeg, False)
	io.imsave("rotated3/"+str(i)+".png", rotated)
	
	#using the OCR
	gifTempFile = "o1.gif"
	tifTempFile = "o2.tif"
	im = Image.open('rotated3/'+str(i)+'.png')
	im.save(gifTempFile, "GIF")
	original = Image.open(gifTempFile)
	bg = original.resize(im.size, Image.NEAREST)
	bg.save(tifTempFile)

	pixImage=tesseract.pixRead(tifTempFile)
	api.SetImage(pixImage)
	outText=api.GetUTF8Text()
	outText = outText.replace("\n","")
	outText = outText.replace("\t","")
	conf=api.MeanTextConf()

	f.write(str(i) + "\t" + outText + "\t" + str(conf)+"\n")

f.close()
api.End()

'''
mask = mask * 255
io.imsave('bing.png', mask)

block_size = 40
binary_adaptive = filters.threshold_adaptive(image, block_size, offset=60)
io.imsave('bin.png', binary_adaptive*100)
'''
