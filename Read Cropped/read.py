import multiprocessing
import cv2 as cv
import numpy as np
import re
import sys
from scipy import ndimage
#from skimage import measure
import pandas as pd
import os
from Operations import addBorders, find_contours, process_kwh, process_numb, resize_kwh, rotate, create_final_answer
import easyocr
from multiprocessing import Lock
import time 



def read_kwh(img):
	#resize the image 
	img = resize_kwh(img, 1400)
	#perform all the processing on the image
	processed_img = process_kwh(img)
	# initialize ocr 
	reader = easyocr.Reader(['en'])
	#find the contours inside the processed image 
	sorted_contours = find_contours(processed_img)
	#drawing White contours above small noise in the image 
	height, width = img.shape[0], img.shape[1]
	for count, cnt in enumerate(sorted_contours):
		x,y,w,h = cv.boundingRect(cnt)
		area = h*w
		area_perc = (area/(height*width))*100
		if(area_perc<0.6):
			cv.drawContours(processed_img, [cnt], -1, (255,255,255), -1)

	#rotate the image 
	rotated_img = rotate(processed_img)

	#finding the contours on the rotated image 
	sorted_contours = find_contours(rotated_img)

	#------ crop the image where there is only the number, after rotating------
	min_y = sys.maxsize
	max_y = 0
	min_x = sys.maxsize
	last_x = -100
	max_height = 0
	pnoe = dict()
	height, width = rotated_img.shape

	for cnt in sorted_contours:
		x,y,w,h = cv.boundingRect(cnt)
		hei_h = height/float(h)
		ratio = h / float(w)
		wid_w = width / float(w)
		cnt_area = h*w
		area_perc = (cnt_area/(height*width))*100
		
		# only let the numbers pass to the ocr by setting contour size criteria             
		if (7.4>hei_h>1.1) and (3.7>ratio>1.1) and (38>wid_w>9) and (5.5>area_perc>0.7):
			#check if the same number or a 0 inside a 0 or two numbers on top of each other 
			if  x-last_x < 50:
				last_x = x 
				continue
			# Collect the locations inside the image where the numbers are 
			if(y<min_y):
				min_y = y
			if(y>max_y):
				max_y = y
			if(max_height<h):
				max_height = h
			if(x<min_x):
				min_x = x

			numb = rotated_img[y-1:y+h+1, x-1:x+w+1]
			# add white border arround the image 
			numb = process_numb(numb)
			# pass the cropped numb to
			numb = cv.bitwise_not(numb, 1)
			# read the number using easyOcr
			easynumb = reader.readtext(numb, allowlist ='0123456789')
		
			if(len(easynumb) != 0):
				last_x = x
				pnoe[x] = [easynumb[0][1], easynumb[0][2]]

	# Sort the numbers using the x axis 
	pnoe_keys = pnoe.items()
	sorted_pnoe = sorted(pnoe_keys)
	
	#----- croping the image using the right dimensions -----#
	# cropping the width of the photo here, is made from the beginning of the original rotated photo till the end. min_x-10:max_x+max_width+10
	cropped_image = rotated_img[min_y:max_y+max_height, min_x-5:width-int(1.5*min_x)+5]
	# add white border 
	cropped_image = addBorders(cropped_image)
	# Add an open morphozlogy to the image 
	rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
	cropped_image = cv.morphologyEx(cropped_image, cv.MORPH_OPEN, rect_kern, iterations=1)
	cropped_image = cv.medianBlur(cropped_image, 5)
	#----- croping the image using the right dimensions ---end---#

    #----- Read the whole image -----#
	# Try reading the whole image 
	# invert the image colors 
	bit_crp = cv.bitwise_not(cropped_image, 1)
	# Read the image using easyocr
	easynumb = reader.readtext(bit_crp, allowlist ='0123456789')
	new_whole = dict()
	for pred in easynumb:
		new_whole[pred[0][0][0] + min_x-5] = [pred[1], pred[2]]

	# Sort the numbers using the x axis 
	new_whole_keys = new_whole.items()
	sorted_new_whole = sorted(new_whole_keys)
	#----- Read the whole image ---end---#


	# finding the contours of the roated cropped image 
	sorted_contours = find_contours(cropped_image)
	pnce = dict()
	last_x = -150

	for count, cnt in enumerate(sorted_contours):
		x,y,w,h = cv.boundingRect(cnt)
		height, width = cropped_image.shape
		hei_h = height/float(h)
		ratio = h / float(w)
		wid_w = width / float(w)
		cnt_area = h*w
		area_perc = (cnt_area/(height*width))*100

		if (3>hei_h>1.05) and (3.6>ratio>1.3) and (35>wid_w>7) and (8>area_perc>1.1):
			#check if the same number or a 0 inside a 0 or two numbers on top of each other 
			if  x-last_x < 50:
				last_x = x 
				continue
			numb = cropped_image[y:y+h, x:x+w]
			numb = process_numb(numb)

			bit_crp = cv.bitwise_not(numb, 1)
			easynumb = reader.readtext(bit_crp, allowlist ='0123456789')
			for pred in easynumb:
				pnce[x+min_x-5] = [easynumb[0][1], easynumb[0][2]]
				last_x = x

	# Sort the numbers using the x axis 
	pnce_keys = pnce.items()
	sorted_pnce = sorted(pnce_keys)


	return create_final_answer(sorted_new_whole, sorted_pnce, sorted_pnoe)


def parallel_exec(cluster, start, lock):
	print(str(start) +" : "  +  str(len(cluster)))

	for img in cluster:
		image = cv.imread('crop/'+img)
		meter = read_kwh(image)
		print(img+": ") 
		lock.acquire()
		with open('output.txt', 'a') as out:
			meter  = img + "," + str(meter)
			out.write(meter+'\n')
			out.close()
		lock.release()
	

if __name__ == '__main__':
	images = os.listdir('crop')
	processes = []
	threads = 12
	lock = Lock()
	chunks = len(images)/threads

	for i in range(1,threads+1):
		start = int(chunks*(i-1))+1
		end = int(chunks*i)
		cluster  =  images[start:end+1]
		p = multiprocessing.Process(target=parallel_exec, args=(cluster, start, lock))
		processes.append(p)
		p.start()
	
	for process in processes:
		process.join()
	
	os.rename(r'output.txt',r'output.csv')


