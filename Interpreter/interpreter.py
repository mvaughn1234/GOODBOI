# import easygopigo3 as gpg3
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
# import picamera

class DieReader():
    # def removeBlob(self,img):
    #     #find all your connected components (white blobs in your image)
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        
    #     sizes = stats[1:, -1]; nb_components = nb_components - 1

    #     # minimum size of particles we want to keep (number of pixels)
        
    #     min_size = 500  

    #     #your answer image
    #     image = np.zeros((output.shape))
    #     #for every component in the image, you keep it only if it's above min_size
    #     for i in range(0, nb_components):
    #         if sizes[i] >= min_size:
    #             image[output == i + 1] = 255

    #     return image
    def harris_corner(self,img_ref,img_crop):
        gray = np.float32(img_ref)
        dst = cv2.cornerHarris(gray,50,15,0.1)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        dst = dst*255.0/dst.max()
        # Threshold for an optimal value, it may vary depending on the image.

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
        print('min/max matrix: [{},{},{},{}]'.format(minr,maxr,minc,maxc))
        img_cropped = img_crop[minr:maxr,minc:maxc]

        msc.imsave('test_out/corners.png',dst)

        return img_cropped

    # Given: picture taken from picam of what should be a dice
    # inside a red border.
    # Output: img cropped to just red border
    def extract_die(self,img):
    	# Threshhold to red
    	# find corners
    	# crop
    	# find circles
    	# interpret number

        # Color space conversion
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ysize = img_gray.shape[0]
        xsize = img_gray.shape[1]

        #Mask Params
        low_red = np.array([100, 0, 0])
        high_red = np.array([255, 50, 50])
        mask_red = cv2.inRange(img, low_red, high_red)
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
        msc.imsave('test_out/orig.png',img)
        msc.imsave('test_out/mask_red.png',mask_red)
        msc.imsave('test_out/mask_onimage.png',mask_onimage)
        msc.imsave('test_out/gray_blur.png',gray_blur)
        msc.imsave('test_out/img_postthresh.png',img_postthresh)
        msc.imsave('test_out/img_edge.png',img_edge)

        img_cropped = self.harris_corner(img_postthresh,img)
        msc.imsave('test_out/img_cropped.png',img_cropped)
        return img_cropped

    def integral(self,im1,integral):
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
        die = self.extract_die(img)
        die = cv2.cvtColor(die, cv2.COLOR_RGB2GRAY)
        h,w = die.shape
        print(h/w)
        if h/w < 0.5 or h/w > 2:
            return -1
        die = cv2.resize(die,(480,480))
        die = 255 - die
        die_integral = np.zeros((481,481),dtype='float64')
        num = self.integral(die,die_integral)
        msc.imsave('test_out/t1.png',die_integral*255/die_integral.max())
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
        response = response*255/response.max()
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
        print(np.sum(response*1.0/255))


reader = DieReader()
img = cv2.imread('Die_Test/4.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w,d = img.shape
padding_h = 500
padding_w = 500
padded = np.ones((h+2*padding_h,w+2*padding_w,d),dtype=img.dtype)*255
shift_h = int(0)
shift_w = int(w/2)
padded[padding_h:int(h-shift_h+padding_h),padding_w:int(w-shift_w+padding_w)] = img[shift_h:h,shift_w:w]
# for row in padded:
#     for pixel in row:
#         val = np.random.rand()
#         if val > 0.99:
#             pixel[0] = 255
#             pixel[1] = 0
#             pixel[2] = 0
msc.imsave('test_out/padded.png',padded)

reader.get_num(padded)