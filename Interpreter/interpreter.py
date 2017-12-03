# import easygopigo3 as gpg3
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
# import picamera

def removeBlob(img):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    
    min_size = 500  

    #your answer image
    image = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            image[output == i + 1] = 255

    return image

def harris_corner(img_ref,img_crop):
	gray = np.float32(img_ref)
	dst = cv2.cornerHarris(gray,50,15,0.1)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	dst = dst*255/dst.max()
	# Threshold for an optimal value, it may vary depending on the image.

	h,w = dst.shape
	corners = np.array([[[h,w],[h,0]],[[0,w],[0,0]]])
	r = 0
	c = 0
	for p in np.nditer(dst):
	    # print('r:{} c:{} p:{}'.format(r,c,p))
	    if p > 25:
	        if r <= corners[0,0,0] and c <= corners[0,0,1]:
	            corners[0,0,0] = r
	            corners[0,0,1] = c
	        if r <= corners[0,1,0] and c >= corners[0,1,1]:
	            corners[0,1,0] = r
	            corners[0,1,1] = c
	        if r >= corners[1,0,0] and c <= corners[1,0,1]:
	            corners[1,0,0] = r
	            corners[1,0,1] = c
	        if r >= corners[1,1,0] and c >= corners[1,1,1]:
	            corners[1,1,0] = r
	            corners[1,1,1] = c

	    if(c == w-1):
	        r = r+1
	        c = 0
	    else:
	        c = c + 1

	minr = min(corners[0,0,0],corners[0,1,0],corners[1,0,0],corners[1,1,0])
	minc = min(corners[0,0,1],corners[0,1,1],corners[1,0,1],corners[1,1,1])
	maxr = max(corners[0,0,0],corners[0,1,0],corners[1,0,0],corners[1,1,0])
	maxc = max(corners[0,0,1],corners[0,1,1],corners[1,0,1],corners[1,1,1])

	img_cropped = img_crop[minr:maxr,minc:maxc]

	cv2.imwrite('test_out/corners.png',dst)

	return img_cropped

# Given: picture taken from picam of what should be a dice
# inside a red border.
# Output: img cropped to just red border
def extract_die(img):
	# Threshhold to red
	# find corners
	# crop
	# find circles
	# interpret number

    # Color space conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ysize = img_gray.shape[0]
    xsize = img_gray.shape[1]

    #Mask Params
    low_red = np.array([100, 0, 0])
    high_red = np.array([255, 50, 50])
    mask_red = cv2.inRange(img_rgb, low_red, high_red)
    mask_onimage = cv2.bitwise_and(img_gray, mask_red)

    #Blur and thresholding
    gray_blur = cv2.GaussianBlur(mask_onimage, (5,5), 0)
    ret, img_postthresh = cv2.threshold(mask_onimage, 50, 255, cv2.THRESH_BINARY)

    #Blob Removal
    # imageDeblob = removeBlob(img_postthresh)

    #Canny Edge
    edge_low = 50
    edge_high = 200
    img_edge = cv2.Canny(img_postthresh, edge_low, edge_high)

    #Display Features
    msc.imsave('test_out/gray.png',img_gray)
    msc.imsave('test_out/orig.png',img_rgb)
    msc.imsave('test_out/mask_red.png',mask_red)
    msc.imsave('test_out/mask_onimage.png',mask_onimage)
    msc.imsave('test_out/gray_blur.png',gray_blur)
    msc.imsave('test_out/img_postthresh.png',img_postthresh)
    msc.imsave('test_out/img_edge.png',img_edge)

    img_rgb_cropped = harris_corner(img_postthresh,img_rgb)
    msc.imsave('test_out/img_cropped.png',img_rgb_cropped)

img = cv2.imread('Die_Test/1.png')
h,w,d = img.shape
padded = np.ones((h+100,w+100,d),dtype=img.dtype)*255
padded[50:h+50,50:w+50,d] = img
msc.imsave('padded.png',padded)
extract_die(padded)