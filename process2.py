# This process just runs the OCR

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
#from pytesser import *
import tesseract

api = tesseract.TessBaseAPI()
api.Init("/usr/share/tesseract-ocr","eng",tesseract.OEM_DEFAULT)
#api.SetPageSegMode(tesseract.PSM_AUTO)
api.SetPageSegMode(tesseract.PSM_SINGLE_WORD)
api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

for i in range(2000):
	gifTempFile = "p1.gif"
	tifTempFile = "p2.tif"
	im = Image.open('newJob/results/'+str(i)+'.png')

	im.save(gifTempFile, "GIF")
	original = Image.open(gifTempFile)
	bg = original.resize(im.size, Image.NEAREST)
	bg.save(tifTempFile)


	pixImage = tesseract.pixRead(tifTempFile)
	api.SetImage(pixImage)
	outText = api.GetUTF8Text()
	outText = outText.replace("\n","")
	outText = outText.replace("\t","")
	conf = api.MeanTextConf()
	print(str(i) + "\t" + outText + "\t" + str(conf))

api.End()
'''
nx, ny = im.size
im2 = im.resize((int(nx*5), int(ny*5)), Image.BICUBIC)
im2.save("temp2.png")
#enh = ImageEnhance.Contrast(im)
#enh.enhance(1.3).show("30% more contrast")
 
imgx = Image.open('temp2.png')
imgx = imgx.convert("RGBA")
pix = imgx.load()
maxval = 0
minval = 1<<24
pixset = set()
alphaset = set()
for y in range(imgx.size[1]):
	for x in range(imgx.size[0]):
		pixval = ((pix[x, y][0] *256 + pix[x, y][1] )*256 + pix[x, y][2])
		pixset.add(pixval)
		alphaset.add(pix[x, y][3])
		maxval = max(maxval, pixval)
		minval = min(minval, pixval)
		#if pixval < 16777215:
		if pix[x, y] == (0, 0, 0, 255):
			pix[x, y] = (0, 0, 0, 255)
		else:
			pix[x, y] = (255, 255, 255, 255)

print(str(sorted(pixset)))
print(str(sorted(alphaset)))

imgx.save("bw.gif", "GIF")
original = Image.open('bw.gif')
bg = original.resize(im.size, Image.NEAREST)
ext = ".tif"
bg.save("input-NEAREST" + ext)
'''
