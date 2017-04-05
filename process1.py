# This process maximizes the entropy

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
import matplotlib.pyplot as plt
import math

#showProcess = True
showProcess = False
NUMBER_OF_IMAGES = 2000

showf = False

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


	
def ternary_search(start, end, points, hull, me, maxError):
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
		
		distA = getDistance(points[hull.vertices], an, me)
		evalA = getEntropy(points, an, me, distA)

		distB = getDistance(points[hull.vertices], bn, me)
		evalB = getEntropy(points, bn, me, distB)
		
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
	
	vals = []
	for x in range(3, len(mask) - 3):
		for y in range(3, len(mask[x]) - 3):
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
	# two repetitions given a function with two peaks
	MAX_ERROR = 0.00001 #max allowed error, aprox. 0.005 degrees
	(val1, eval1) = ternary_search(0, math.pi, points, hull, me, MAX_ERROR)
	(val2, eval2) = ternary_search(0, val1, points, hull, me, MAX_ERROR)
	(val3, eval3) = ternary_search(val1, math.pi, points, hull, me, MAX_ERROR)
	
	a = val1
	evalA = eval1
	if eval2 > eval1:
		(a, evalA)  = (val2, eval2)
	else:
		if eval3 > eval1:
			(a, evalA) = (val3, eval3)
	
	adeg = math.degrees(a)
	if showProcess:
		print("angle found: " + str(a))
		print("angle found(degrees): " + str(adeg))
		print("max entropy found: " + str(evalA))
	else:
		print(str(adeg - 90.0))
		
	#remove borders to mask
	maskN = mask[4:-4,4:-4]
	rotated = rotate(maskN, 90.0-adeg, False, (me[1] - 4,me[0] - 4 ))
	#rotated = rotate(maskN, 90.0-adeg, False)
	io.imsave("rotated/"+str(i)+".png", rotated)

'''
mask = mask * 255
io.imsave('bing.png', mask)

block_size = 40
binary_adaptive = filters.threshold_adaptive(image, block_size, offset=60)
io.imsave('bin.png', binary_adaptive*100)
'''
