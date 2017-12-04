# import easygopigo3 as gpg3
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
# import picamera

class DieReader():
    def harris_corner(self,img_ref,img_crop):
        # print('corner')
        gray = np.float32(img_ref)
        dst = cv2.cornerHarris(gray,5,1,0.01)
        has_corner1 = 0
        has_corner2 = 0
        has_corner3 = 0
        has_corner4 = 0

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        dst = dst-dst.min()
        if dst.max() > 0:
            dst = (dst/dst.max())*255
        # print(dst)
        # print(dst.max())
        # print(dst.min())
        # Threshold for an optimal value, it may vary depending on the image.

        h,w = dst.shape
        corners = np.array([[[h,w],[h,0]],[[0,w],[0,0]]])
        r = 0
        c = 0
        # print('iter')
        # print(dst.shape)
        dstmin = dst.min()+0.1*(255-dst.min())
        for p in np.nditer(dst):
            # print('r:{} c:{} p:{}'.format(r,c,p))
            if p > dstmin:
                if np.sqrt(r**2+c**2) <= np.sqrt(corners[0,0,0]**2+corners[0,0,1]**2):
                    corners[0,0,0] = r
                    corners[0,0,1] = c
                    has_corner1 = 1
                if np.sqrt(r**2 + (c-w)**2) <= np.sqrt(corners[0,1,0]**2 + (corners[0,1,1]-w)**2):
                    corners[0,1,0] = r
                    corners[0,1,1] = c
                    has_corner2 = 1
                if np.sqrt((r-h)**2+c**2) <= np.sqrt((corners[1,0,0]-h)**2 + (corners[1,0,1])**2):
                    corners[1,0,0] = r
                    corners[1,0,1] = c
                    has_corner3 = 1
                if np.sqrt((r-h)**2+(c-w)**2) <= np.sqrt((corners[1,1,0]-h)**2 + (corners[1,1,1]-w)**2):
                    corners[1,1,0] = r
                    corners[1,1,1] = c
                    has_corner4 = 1

            if(c == w-1):
                r = r+1
                c = 0
            else:
                c = c + 1
        # print('iter2')
        # print(corners)
        minr = min(corners[0,0,0],corners[0,1,0],corners[1,0,0],corners[1,1,0])
        minc = min(corners[0,0,1],corners[0,1,1],corners[1,0,1],corners[1,1,1])
        maxr = max(corners[0,0,0],corners[0,1,0],corners[1,0,0],corners[1,1,0])
        maxc = max(corners[0,0,1],corners[0,1,1],corners[1,0,1],corners[1,1,1])
        dw = int(0.05*(maxr-minr))
        dh = int(0.05*(maxc-minc))
        # print('min/max matrix: [{},{},{},{}]'.format(minr,maxr,minc,maxc))
        img_cropped = img_crop[(minr+dh):(maxr-dh),(minc+dw):(maxc-dw)]
        # print(dst)
        msc.imsave('test_out/corners.png',dst)

        if(has_corner1 == has_corner2 == has_corner3 == has_corner4 == 1):
            msc.imsave('test_out/img_cropped.png',img_cropped)
            return img_cropped
        else:
            return -1

    def dominant(self,img):
        dom = np.copy(img)
        h,w,d = img.shape
        for row in dom:
            for pixel in row:
                i = np.argmax(pixel)
                if i == 0:
                    if(pixel[0]/pixel[1] > 1.1 and pixel[0]/pixel[2] > 1.1):
                        pixel[1] = 1
                        pixel[2] = 1
                elif i == 1:
                    if(pixel[1]/pixel[0] > 1.1 and pixel[1]/pixel[2] > 1.1):
                        pixel[0] = 1
                        pixel[2] = 1
                elif i == 2:
                    if(pixel[2]/pixel[1] > 1.1 and pixel[2]/pixel[0] > 1.1):
                        pixel[0] = 1
                        pixel[1] = 1
                # print(pixel)
        return dom
    # Given: picture taken from picam of what should be a dice
    # inside a red border.
    # Output: img cropped to just red border
    def extract_die(self,img):
        # print('extract')
    	# Threshhold to red
    	# find corners
    	# crop
    	# find circles
    	# interpret number

        # Color space conversion
        dom = self.dominant(img)
        msc.imsave('test_out/dom.png',dom)
        # print(dom)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray_intensity_scale = int(255/img_gray.max())
        # img_gray *= gray_intensity_scale
        ysize = img_gray.shape[0]
        xsize = img_gray.shape[1]

        #Mask Params
        low_red = np.array([120, 0, 0])
        high_red = np.array([255, 120, 120])
        mask_red = cv2.inRange(dom, low_red, high_red)
        # print(mask_red.shape)
        # print(img_gray.shape)
        msc.imsave('test_out/mask_red.png',mask_red)
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
        msc.imsave('test_out/mask_onimage.png',mask_onimage)
        msc.imsave('test_out/gray_blur.png',gray_blur)
        msc.imsave('test_out/img_postthresh.png',img_postthresh)
        msc.imsave('test_out/img_edge.png',img_edge)
        # return
        img_cropped = self.harris_corner(img_postthresh,img)
        return img_cropped

    def integral(self,im1,integral):
        # print('integral')
        h,w = im1.shape
        r = 0; c = 0;
        s = 0
        for p in np.nditer(im1):
            s = s + p
            integral[r+1,c+1] = s + integral[r,c+1]

            if(c == w-1):
                c = 0
                r = r+1
                s = 0
            else:
                c = c + 1

    def get_num(self,img):
        # print('getnum')
        # print(img)
        die = self.extract_die(img)
        if type(die) == type(1):
            return -1
        die = cv2.cvtColor(die, cv2.COLOR_RGB2GRAY)
        msc.imsave('test_out/check1.png',die)
        die = die-die.min()
        msc.imsave('test_out/check2.png',die)
        die = (die/die.max())*255
        msc.imsave('test_out/check3.png',die)
        h,w = die.shape
        # print(h/w)
        if h/w < 0.5 or h/w > 2:
            return -1
        die = cv2.resize(die,(480,480))
        msc.imsave('test_out/check4.png',die)
        die = 255 - die
        die_integral = np.zeros((481,481),dtype='float64')
        self.integral(die,die_integral)
        # print(die)
        # print(die_integral)
        die_integral = (die_integral/die_integral.max())*255
        msc.imsave('test_out/die_inv.png',die)
        msc.imsave('test_out/t1.png',die_integral)
        #   0 0 0 0
        #   0 1 2 3  ->   D B
        #   0 4 5 6  ->   C A
        #   0 7 8 9
        # Sum = A-B-C+D
        p1 = die_integral[160,160]
        p2 = die_integral[160,320]
        p3 = die_integral[160,480]
        p4 = die_integral[320,160]
        p5 = die_integral[320,320]
        p6 = die_integral[320,480]
        p7 = die_integral[480,160]
        p8 = die_integral[480,320]
        p9 = die_integral[480,480]
        n1 = p1
        n2 = p2-p1
        n3 = p3-p2
        n4 = p4-p1
        n5 = p5-p4-p2+p1 
        n6 = p6-p5-p3+p2
        n7 = p7-p4
        n8 = p8-p7-p5+p4
        n9 = p9-p8-p6+p5
        template = np.ones((160,160),dtype = 'float64')
        demo = np.zeros((480,480),dtype='float64')
        response = np.array([[n1,n2,n3],[n4,n5,n6],[n7,n8,n9]])
        response = response-response.min()
        response = response*255/response.max()
        # print(response)
        demo[0:160,0:160] = template*int(response[0,0])
        demo[0:160,160:320] = template*int(response[0,1])
        demo[0:160,320:480] = template*int(response[0,2])
        demo[160:320,0:160] = template*int(response[1,0])
        demo[160:320,160:320] = template*int(response[1,1])
        demo[160:320,320:480] = template*int(response[1,2])
        demo[320:480,0:160] = template*int(response[2,0])
        demo[320:480,160:320] = template*int(response[2,1])
        demo[320:480,320:480] = template*int(response[2,2])
        msc.imsave('test_out/demo_Test.png',demo)
        num = 0
        for ele in np.nditer(response):
            if ele > 150:
                num = num + 1
        # print(np.sum(response*1.0/255))
        return num


def test(pic,padding_h,padding_w,shift_h,shift_w,shift_left,shift_up,noise):
    reader = DieReader()
    img = cv2.imread('Die_Test/'+pic)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w,d = img.shape
    img = cv2.resize(img,(480,int(h*480/w)))
    msc.imsave('test_out/orig.png',img)

    h,w,d = img.shape
    padded = np.ones((h+2*padding_h,w+2*padding_w,d),dtype=img.dtype)*255
    shift_h = int(shift_h)
    shift_w = int(shift_w)
    padded[padding_h:int(h-shift_h+padding_h),padding_w:int(w-shift_w+padding_w)] = img[shift_h:h,shift_w:w]
    hp,wp,dp = padded.shape
    shift_left = int(shift_left)
    shift_up = int(shift_up)
    padded2 = padded[shift_left:hp,shift_up:wp,:]
    padded = padded2
    # padded = cv2.resize(padded,(480,480))
    for row in padded:
        for pixel in row:
            val = np.random.rand()
            if val > noise:
                pixel[0] = 255
                pixel[1] = 0
                pixel[2] = 0

    msc.imsave('test_out/padded.png',padded)
    print(reader.get_num(padded))
    # print(reader.get_num(img))

test('test5.jpg',0,0,0,0,0,0,1.2)