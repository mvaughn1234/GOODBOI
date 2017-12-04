import cv2
import numpy as np
import scipy.misc as msc

def harris_corner(img_ref,img_crop):
    # print('corner')
    gray = np.float32(img_ref)
    dst = cv2.cornerHarris(gray,10,3,0.01)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    dst = dst-dst.min()
    dst = (dst*255.0/dst.max())
    print(dst)
    # print(dst.max())
    # print(dst.min())
    # Threshold for an optimal value, it may vary depending on the image.

    h,w = dst.shape
    corners = np.array([[[h,w],[h,0]],[[0,w],[0,0]]])
    r = 0
    c = 0
    # print('iter')
    # print(dst.shape)
    dstmin = dst.min()+50
    for p in np.nditer(dst):
        # print('r:{} c:{} p:{}'.format(r,c,p))
        if p > dstmin:
            if r <= corners[0,0,0] and c <= corners[0,0,1]:
                corners[0,0,0] = r
                corners[0,0,1] = c
            if r <= corners[0,1,0] and c >= corners[0,1,1]:
                corners[0,1,0] = r
                corners[0,1,1] = c
            if r >= corners[1,0,0] and c <= corners[1,0,1]:
                corners[1,0,0] = r
                corners[1,0,1] = c
            if r >= corners[1,1,0] or c >= corners[1,1,1]:
                corners[1,1,0] = r
                corners[1,1,1] = c

        if(c == w-1):
            r = r+1
            c = 0
        else:
            c = c + 1
    # print('iter2')
    print(corners)
    minr = min(corners[0,0,0],corners[0,1,0],corners[1,0,0],corners[1,1,0])
    minc = min(corners[0,0,1],corners[0,1,1],corners[1,0,1],corners[1,1,1])
    maxr = max(corners[0,0,0],corners[0,1,0],corners[1,0,0],corners[1,1,0])
    maxc = max(corners[0,0,1],corners[0,1,1],corners[1,0,1],corners[1,1,1])
    print('min/max matrix: [{},{},{},{}]'.format(minr,maxr,minc,maxc))
    img_cropped = img_crop[minr:maxr,minc:maxc]
    # print(dst)
    msc.imsave('corners.png',dst)
    msc.imsave('cropped.png',img_cropped)

    return img_cropped

img = cv2.imread('img_postthresh.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('orig.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
h,w = img.shape
img = cv2.resize(img,(480,int(h*480/w)))
img2 = cv2.resize(img2,(480,int(h*480/w)))
msc.imsave('check1.png',img)
msc.imsave('check2.png',img2)
harris_corner(img,img2)