import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage


def addBorders(img):
	row, col = img.shape[:2]
	mean = 255

	bordersize = 10
	border = cv.copyMakeBorder(
		img,
		top=bordersize,
		bottom=bordersize,
		left=bordersize,
		right=bordersize,
		borderType=cv.BORDER_CONSTANT,
		value=[mean]
	)
	return border

def rotate(closing):
	edges = cv.Canny(closing,50,150,apertureSize = 3)
	lines = cv.HoughLines(edges,1,np.pi/180,200)
	
	if lines is None:
		# no lines were found, the image might be cropped very tightly 
		# return same image 
		return closing
	for rho,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

	cv.line(closing,(x1,y1),(x2,y2),(0,0,255),2)

	return ndimage.rotate(closing, 180*theta/3.1415926-90)

def resize_kwh(img, desired_width):
	# takes an image and a specified width lenght and resize the image accordingly 
	img_height = img.shape[0]
	img_width = img.shape[1]
	scale_percent = (desired_width*100)/img_width

	#calculate the percent of original dimensions
	width = int(img_width * scale_percent / 100)
	height = int(img_height * scale_percent / 100)
	dsize = (width, height)
	#resizing image by the same percentage
	img = cv.resize(img, dsize)
	return img


def process_kwh(img):
	# Pick all the pixels inside the picture that are above 200, they are the glare most probably, and make them all black 
	img[np.where((img>[220,225,230]).all(axis=2))] = [0,0,0]

	#gray scaling
	gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

	#remove noise from photo highly effective
	blur_img = cv.GaussianBlur(gray_img, (5,5), 0)

	# create rectangular kernel for dilation
	rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

	#thresholding the image using otsu method and binary inverse since our background is black and the numbers in white
	ret, thresh = cv.threshold(blur_img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

	#closing is method of thresholding the image too, it usually more clean than the threshold method by itself
	closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rect_kern, iterations=2)

	#Adaptive thresholding, used for taking care of the glare
	th2 = cv.adaptiveThreshold(blur_img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,15,2)

	# closing morphology of the adaptive thresholded image
	closing_ADT = cv.morphologyEx(th2, cv.MORPH_CLOSE, rect_kern, iterations=1)

	#Replacing the pixels in the original closing image where there is w white pixel in the adaptive thresholded image
	closing[np.where(closing_ADT == 255)] = 255

	return closing

def process_numb(numb):
	numb = addBorders(numb)
	#denoising
	numb = cv.medianBlur(numb, 5)
	# open morphology to fill inside the numbers
	rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
	numb = cv.morphologyEx(numb, cv.MORPH_OPEN, rect_kern, iterations=2)

	return numb


def find_contours(img):
	# finding the contours of the roated cropped image 
	contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	#CONTOUR IS A PYTHON LIST OF ALL CONTOURS IN THE IMAGE. PUT IN A NUMPY ARRAY
	sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

	return sorted_contours

def pick(dic):
	count = dict()
	for e in dic[1]:
		if e[0][:1] in count:
			count[e[0][:1]] += e[1]
		else: count[e[0][:1]] = e[1]

	# pick the highest
	highest = 0
	num = None
	for c in count:
		if count[c] > highest:
			num = c
			highest = count[c]	

	return str(num)


def create_final_answer(whole, crop, org):
	combined = dict()
	avg = 0
	if len(org) > 0:
		o_avg = 0
		last_o = org[0][0]
		for o in org:
			o_avg += o[0] - last_o
			last_o = o[0]
			combined[o[0]] = [o[1]]
		avg += o_avg/len(org)

	if len(whole) > 0:
		w_avg = 0
		last_w = whole[0][0]
		for w in whole:
			w_avg += w[0]-last_w
			last_w = w[0]
			flag = 0
			for com in combined.copy():
				if abs(w[0]-com) < 70:
					flag = 1
					combined[com].append(w[1])
			if flag == 0:
				combined[w[0]] = [w[1]]

	if len(crop) > 0:
		c_avg = 0
		last_c = crop[0][0]
		for c in crop:
			c_avg += c[0]-last_c
			last_c = c[0]
			flag = 0
			for com in combined.copy():
				if abs(c[0]-com) < 70:
					flag = 1
					combined[com].append(c[1])
			if flag == 0:
				combined[c[0]] = [c[1]]
		avg +=  c_avg/len(crop)
	avg = avg/2
	combined_keys = combined.items()
	combined = sorted(combined_keys)
	final = ""

	last_passed =-500
	found = 0
	for c in combined:
		if c[0] - last_passed > avg-45 and found < 6:
			found += 1
			final += pick(c)
			last_passed = c[0]

		
	return final