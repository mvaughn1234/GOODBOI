import cv2
import numpy as np

def harris_corner(self,img_ref,img_crop):
	gray = cv2.cvtColor(img_ref,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,50,15,0.1)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	dst = dst*255/dst.max()
	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]

	h,w = dst.shape
	corners = np.array([[[h,w],[h,0]],[[0,w],[0,0]]])
	r = 0
	c = 0
	for p in np.nditer(dst):
	    # print('r:{} c:{} p:{}'.format(r,c,p))
	    if p > 50:
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

	cropped = dst[minr:maxr,minc:maxc]
	img_cropped = img[minr:maxr,minc:maxc]

	cv2.imwrite('test_out/corners.png',dst)
	cv2.imwrite('test_out/corners_cropped.png',cropped)
	cv2.imwrite('test_out/img.png',img)
	cv2.imwrite('test_out/img_cropped.png',img_cropped)

	return img_cropped