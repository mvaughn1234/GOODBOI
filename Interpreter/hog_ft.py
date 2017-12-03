import numpy as np
from numpy import arctan2
import matplotlib.pyplot as plt
import scipy.misc as msc
import sys
import cv2
import pickle
import inspect
import unittest

class HOG_Descriptor:
	def __init__(self):
		pass

	def filter(self,f_matrix,image):
		h,w = image.shape #Get height and width of image
		# print('shape: {},{}'.format(h,w))
		smooth = np.zeros((h,w),dtype = 'float32')
		r=0; c=0; #For keeping track of which row,col is being examined
		for x in np.nditer(image): #use nditer to iterate through every element
			#3 cases: (top/left),middle,(bottom/right)
			# print('({},{}): {}'.format(c,r,x))
			#First get ygradient at current element
			#121
			#242
			#121
			if(r == 0):
				if(c == 0):
					smooth[r,c] = min(f_matrix[0,0]*image[r,c]+f_matrix[0,1]*image[r,c]+f_matrix[0,2]*image[r,c+1]+\
								f_matrix[1,0]*image[r,c]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c+1]+\
								f_matrix[2,0]*image[r+1,c]+f_matrix[2,1]*image[r+1,c]+f_matrix[2,2]*image[r+1,c+1],255)
				elif(c == w-1):
					smooth[r,c] = min(f_matrix[0,0]*image[r,c-1]+f_matrix[0,1]*image[r,c]+f_matrix[0,2]*image[r,c]+\
								f_matrix[1,0]*image[r,c-1]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c]+\
								f_matrix[2,0]*image[r+1,c-1]+f_matrix[2,1]*image[r+1,c]+f_matrix[2,2]*image[r+1,c],255)
				else:
					smooth[r,c] = min(f_matrix[0,0]*image[r,c-1]+f_matrix[0,1]*image[r,c]+f_matrix[0,2]*image[r,c+1]+\
								f_matrix[1,0]*image[r,c-1]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c+1]+\
								f_matrix[2,0]*image[r+1,c-1]+f_matrix[2,1]*image[r+1,c]+f_matrix[2,2]*image[r+1,c+1],255)
			elif(r == h-1):
				if(c == 0):
					smooth[r,c] = min(f_matrix[0,0]*image[r-1,c]+f_matrix[0,1]*image[r-1,c]+f_matrix[0,2]*image[r-1,c+1]+\
								f_matrix[1,0]*image[r,c]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c+1]+\
								f_matrix[2,0]*image[r,c]+f_matrix[2,1]*image[r,c]+f_matrix[2,2]*image[r,c+1],255)
				elif(c == w-1):
					smooth[r,c] = min(f_matrix[0,0]*image[r-1,c-1]+f_matrix[0,1]*image[r-1,c]+f_matrix[0,2]*image[r-1,c]+\
								f_matrix[1,0]*image[r,c-1]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c]+\
								f_matrix[2,0]*image[r,c-1]+f_matrix[2,1]*image[r,c]+f_matrix[2,2]*image[r,c],255)
				else:
					smooth[r,c] = min(f_matrix[0,0]*image[r-1,c-1]+f_matrix[0,1]*image[r-1,c]+f_matrix[0,2]*image[r-1,c+1]+\
								f_matrix[1,0]*image[r,c-1]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c+1]+\
								f_matrix[2,0]*image[r,c-1]+f_matrix[2,1]*image[r,c]+f_matrix[2,2]*image[r,c+1],255)
			else:
				if(c == 0):
					smooth[r,c] = min(f_matrix[0,0]*image[r-1,c]+f_matrix[0,1]*image[r-1,c]+f_matrix[0,2]*image[r-1,c+1]+\
								f_matrix[1,0]*image[r,c]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c+1]+\
								f_matrix[2,0]*image[r+1,c]+f_matrix[2,1]*image[r+1,c]+f_matrix[2,2]*image[r+1,c+1],255)
				if(c == w-1):
					smooth[r,c] = min(f_matrix[0,0]*image[r-1,c-1]+f_matrix[0,1]*image[r-1,c]+f_matrix[0,2]*image[r-1,c]+\
								f_matrix[1,0]*image[r,c-1]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c]+\
								f_matrix[2,0]*image[r+1,c-1]+f_matrix[2,1]*image[r+1,c]+f_matrix[2,2]*image[r+1,c],255)
				else:
					smooth[r,c] = min(f_matrix[0,0]*image[r-1,c-1]+f_matrix[0,1]*image[r-1,c]+f_matrix[0,2]*image[r-1,c+1]+\
								f_matrix[1,0]*image[r,c-1]+f_matrix[1,1]*image[r,c]+f_matrix[1,2]*image[r,c+1]+\
								f_matrix[2,0]*image[r+1,c-1]+f_matrix[2,1]*image[r+1,c]+f_matrix[2,2]*image[r+1,c+1],255)
			if(c == 0):
				c = c + 1
			elif(c == w-1):
				c = 0
				r = r + 1
			else:
				c = c + 1

		return smooth
		pass

	def random_filter(self,image):
		random_filter = np.array([[0,0,0],[0,2,0],[0,0,0]])
		# test = self.filter(random_filter,image)
		test = cv2.filter2D(image,-1,random_filter)
		# print(test)
		return tes

	def smooth(self,image):
		smoothing_filter = np.ones((8,8))/(8*8)
		# smoothing_filter = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
		# smoothing_filter = np.array([[1,1,1],[1,1,1],[1,1,1]])/9
		# smoothing_filter = np.array([[0,0,0],[0,1,0],[0,0,0]])
		# smoothed = self.filter(smoothing_filter,image)
		smoothed = cv2.filter2D(image,-1,smoothing_filter)
		# print(smoothed)
		return smoothed

	def sharpen(self,image):
		# sharpening_filter = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,3,0,0],[0,0,0,0,0],[0,0,0,0,0]])
		sharpening_filter = np.array([[0,0,0],[0,2,0],[0,0,0]])
		smoothing_filter = np.ones((3,3))/(3*3)
		# sharpened = (self.filter(sharpening_filter,image) - self.filter(smoothing_filter,image))
		sharpened = (cv2.filter2D(image,-1,sharpening_filter) - cv2.filter2D(image,-1,smoothing_filter))
		# print(sharpened)
		return sharpened

	#Image must be 2Dimensional.
	#If a RGB image must be examined, do it in 3 function calls.
	def gradient(self,image):
		image = self.smooth(image)
		h,w = image.shape #Get height and width of image
		# print('shape: {},{}'.format(h,w))
		gradx = np.zeros((h,w),dtype = 'int')
		grady = np.zeros((h,w),dtype = 'int')
		r=0; c=0; #For keeping track of which row,col is being examined
		for x in np.nditer(image): #use nditer to iterate through every element
			#3 cases: (top/left),middle,(bottom/right)
			# print('({},{}): {}'.format(c,r,x))
			#First get ygradient at current element
			if(r == 0):
				grady[r,c] = int(image[r+1,c]) - int(image[r,c])
			elif(r == h-1):
				grady[r,c] = int(image[r,c]) - int(image[r-1,c])
			else:
				grady[r,c] = int(image[r+1,c]) - int(image[r-1,c])
			#Second get xgradient at current element
			if(c == 0):
				gradx[r,c] = int(image[r,c+1]) - int(image[r,c])
				c = c + 1
			elif(c == w-1):
				gradx[r,c] = int(image[r,c]) - int(image[r,c-1])
				c = 0
				r = r + 1
			else:
				gradx[r,c] = int(image[r,c+1]) - int(image[r,c-1])
				c = c + 1
		return gradx,grady
		pass


	def gradient_RGB(self,image):
		dim = image.shape
		if(np.size(dim)==3):
			h,w,d = image.shape
			# print('Computing grad, ch1 w:{} h:{}...'.format(w,h))
			final_gradx,final_grady = self.gradient(image[:,:,0])
			# print(final_gradx)
			# print(final_grady)
			for i in range(1,d):
				# print('Computing grad, ch{} w:{} h:{}...'.format(i+1,w,h))
				image_ch = image[:,:,i]
				c = 0; r = 0;
				new_gradx,new_grady = self.gradient(image[:,:,i])
				# print(new_gradx)
				for x in np.nditer(final_gradx):
					# print('i:{} c:{} r:{} old:{} new:{}'.format(i,c,r,x,new_gradx[r,c]))
					if(abs(x) < abs(new_gradx[r,c])):
						final_gradx[r,c] = new_gradx[r,c]
					if(c == w-1):
						c = 0
						r = r+1
					else:
						c = c+1
				c = 0; r = 0;
				# print(new_grady)
				for y in np.nditer(final_grady):
					# print('i:{} c:{} r:{} old:{} new:{}'.format(i,c,r,y,new_grady[c,r]))
					if(abs(y) < abs(new_grady[r,c])):
						final_grady[r,c] = new_grady[r,c]
					if(c == w-1):
						c = 0
						r = r+1
					else:
						c = c+1
			return final_gradx,final_grady
		else:
			return self.gradient(image)
		pass

	def magnitude_orientation(self,gx,gy):
		# print('Computing mag/ori...')
		h,w = gx.shape
		mag = np.zeros((h,w),dtype = 'float32')
		ori = np.zeros((h,w),dtype = 'float32')
		r = 0; c = 0;
		for x in np.nditer(gx):
			mag[r,c] = np.sqrt(pow(gx[r,c],2)+pow(gy[r,c],2))
			ori[r,c] = np.mod(np.arctan2(gy[r,c],gx[r,c])*180/np.pi,180)
			if(c == w-1):
				c = 0
				r = r + 1
			else:
				c = c+1
		return mag,ori
		pass

	def magnitude_orientation_RGB(self,img):
		dim = img.shape
		if(np.size(dim)==3):
			h,w,d = img.shape
			gradx,grady = self.gradient(img[:,:,0])
			final_mag,final_ori = self.magnitude_orientation(gradx,grady)
			for i in range(1,d):
				img_ch = img[:,:,i]
				c = 0; r = 0;
				new_gradx,new_grady = self.gradient(img_ch)
				new_mag,new_ori = self.magnitude_orientation(new_gradx,new_grady)
				for x in np.nditer(final_mag):
					if(x < new_mag[r,c]):
						final_mag[r,c] = new_mag[r,c]
						final_ori[r,c] = new_ori[r,c]
					if(c == w-1):
						c = 0
						r = r+1
					else:
						c = c+1
			return final_mag,final_ori
		else:
			gradx,grady = self.gradient(img)
			final_mag,final_ori = self.magnitude_orientation(gradx,grady)
			return final_mag,final_ori


	def show_cells(self,cell_hist,num_cells_height,num_cells_width,n_bins):
		#0-20,20-40,40-60,60-80,80-100,100-120,120-140,140-160,160-180
		nch = num_cells_height; ncw = num_cells_width; nb = n_bins;
		b1 = [[0,0,0],[255,255,255],[0,0,0]]
		b2 = [[255,0,0],[255,255,255],[0,0,255]]
		b3 = [[255,0,0],[0,255,0],[0,0,255]]
		b4 = [[255,255,0],[0,255,0],[0,255,255]]
		b5 = [[0,255,0],[0,255,0],[0,255,0]]
		b6 = [[0,255,255],[0,255,0],[255,255,0]]
		b7 = [[0,0,255],[0,255,0],[255,0,0]]
		b8 = [[0,0,255],[255,255,255],[255,0,0]]
		b9 = b1
		outim = np.array((nch,ncw),dtype=type(b1))
		for r in range(0,nch):
			for c in range(0,ncw):
				max_bin = 0
				for n in range(0,n_bins):
					if cell_hist[r,c,max_bin] < cell_hist[r,c,n]:
						max_bin = n
				if(max_bin == 0):
					outim[r,c] = b1
				if(max_bin == 1):
					outim[r,c] = b2
				if(max_bin == 2):
					outim[r,c] = b3
				if(max_bin == 3):
					outim[r,c] = b4
				if(max_bin == 4):
					outim[r,c] = b5
				if(max_bin == 5):
					outim[r,c] = b6
				if(max_bin == 6):
					outim[r,c] = b7
				if(max_bin == 7):
					outim[r,c] = b8
				if(max_bin == 8):
					outim[r,c] = b9
		print(outim)
		print(np.vstack(outim))


	def compute(self,img,block_size,cell_size,n_bins,block_stride):
		gx,gy = self.gradient_RGB(img)
		msc.imsave('gx.png',gx)
		msc.imsave('gy.png',gy)
		# print(gx)
		# print(gy)
		mag,ori = self.magnitude_orientation_RGB(img)
		# mag = self.smooth(mag)
		msc.imsave('mag.png',mag)
		# print(mag)
		# print(ori)
		bin_incr = 180/n_bins; bin_start = bin_incr/2;
		h,w = mag.shape
		# print('({},{})'.format(h,w))
		num_cells_width = int(w/cell_size)
		num_cells_height = int(h/cell_size)
		cell_hist = np.zeros((num_cells_height,num_cells_width,n_bins),dtype='float32')
		cellr = 0; cellc = 0;
		r = 0; c = 0;
		# print('_Startup_\nbi:{} bs:{} NCW:{} NCH:{}'.format(bin_incr,bin_start,num_cells_width,num_cells_height))
		# print('_Histogram Cells_\n...')
		for x in np.nditer(gx):
			#ori = bin_incr*correct_bin - bin_start
			#correct_bin = (ori+bin_start)/bin_incr
			#ex 165deg = 8.75th bin is 0.75*mag into 9th bin and 0.25*mag into 8th
			m = mag[r,c]; o = ori[r,c];
			correct_bin = (o+bin_start)/bin_incr
			frac = np.mod(correct_bin,1)
			bin1_index = np.mod(int(correct_bin)-1,n_bins)
			bin2_index = np.mod(int(correct_bin),n_bins)
			bin1_allowance = (1-frac)*m
			bin2_allowance = frac*m
			cell_hist[cellr,cellc,bin1_index] += bin1_allowance;
			cell_hist[cellr,cellc,bin2_index] += bin2_allowance;
			# print('mag:{} ori:{} b1i:{} b2i:{} amtb1:{} amtb2:{}'.format(m,o,bin1_index,bin2_index,bin1_allowance,bin2_allowance))
			# print('c:{} r:{} cellc:{} cellr:{} bin1:{} amt:{} bin2:{} amt:{}'.format(c,r,cellc,cellr,bin1_index,bin1_allowance,bin2_index,bin2_allowance))
			if(c == w-1):
				c = 0
				r = r + 1
			else:
				c = c+1
			cellr = int(r/cell_size)
			cellc = int(c/cell_size)
		# print('Cell Hist:')
		# print(cell_hist)
		num_blocks_width = int((w-block_size)/block_stride)+1
		num_blocks_height = int((h-block_size)/block_stride)+1
		num_cells_per_block = int(block_size/cell_size)
		block_hist = np.zeros((num_blocks_width*num_blocks_height*pow(num_cells_per_block,2)*n_bins),dtype='float32')
		b = 0;
		block_cell_r = 0; block_cell_c = 0; # Local cell index inside block
		global_cell_r = 0; global_cell_c = 0; # Global cell index
		block_r = 0; block_c = 0; # Block index in image
		block_start_ele = 0; # Start element of current block in histogram
		ele = 0; block_hist_sqr_sum = 0;
		# print('_Block Startup_\nNBW:{} NBH:{} NCPB:{}'.format(num_blocks_width,num_blocks_height,num_cells_per_block))
		# print('Histogram Blocks')
		for x in np.nditer(block_hist):
			global_cell_r = int(block_r*(block_stride/cell_size)+block_cell_r)
			global_cell_c = int(block_c*(block_stride/cell_size)+block_cell_c)
			# print('ele:{} GCC:{} GCR:{} b:{}'.format(ele,global_cell_c,global_cell_r,b))
			val = cell_hist[global_cell_r,global_cell_c,b]
			block_hist[ele] = val
			block_hist_sqr_sum += pow(val,2)
			# print('Blockc:{} Blockr:{} val:{} sum:{}'.format(block_c,block_r,val,block_hist_sqr_sum))
			if(np.mod(((ele+1)/n_bins),1)==0):
				# finished current cell's histogram, go to next cell in current bin
				b = 0
				if(np.mod(((block_cell_c+1)/num_cells_per_block),1)==0):
					#End of current row of cells inside block
					block_cell_c = 0
					if(np.mod(((block_cell_r+1)/num_cells_per_block),1)==0):
						#End of current block
						block_cell_r = 0
						# print('normalizing {}:{}'.format(block_start_ele,ele))
						block_hist[block_start_ele:ele+1] = block_hist[block_start_ele:ele+1]/np.sqrt(block_hist_sqr_sum+1e-7)
						# block_hist[block_start_ele:ele+1] = block_start_ele
						block_hist_sqr_sum = 0;
						block_start_ele = ele + 1;
						# print('End Block')
						if(block_c == num_blocks_width-1):
							#End of row of blocks
							block_c = 0
							if(block_r == num_blocks_height-1):
								#End of image
								# print('End Image')
								break
							else:
								#End of row of blocks, not end of image, go to next row of blocks
								block_r = block_r + 1
								# print('End Block Row')
								# print('Start block:({},{})'.format(block_c,block_r))
						else:
							#End of current block, not end of row, go to next block in row
							block_c = block_c + 1
							# print('Start block:({},{})'.format(block_c,block_r))
					else:
						#End of current row of cells inside block, not end of block, go to next row of cells in block
						block_cell_r = block_cell_r + 1
				else:
					#End of current cell's histogram, not end of current row of cells, go to next cell in row
					block_cell_c = block_cell_c + 1
			else:
				#End of nothing, go to next bin in histogram
				b = b + 1
			ele = ele + 1
		# print(block_hist)
		# print('Block Hist (len {}):'.format(len(block_hist)))
		# print(block_hist)
		return block_hist
		pass

# testhog = HOG_Descriptor()
# image = cv2.imread('boats/boat1.jpg')
# image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.resize(image,(640,480))
# # rand = testhog.random_filter(image)
# blurred = testhog.smooth(image)
# sharper = testhog.sharpen(image)
# # sharper = cv2.resize(sharper,(640,480))
# # blurred = cv2.resize(blurred,(640,480))
# msc.imsave('orig.jpg',image)
# # msc.imsave('rand.jpg',rand)
# msc.imsave('blur.jpg',blurred)
# msc.imsave('sharpen.jpg',sharper)
# n_bins = 9;
# block_size = 16
# cell_size = 8
# hog_desc = testhog.compute(image,block_size,cell_size,n_bins,block_stride =2*cell_size)
# print('{}'.format(len(hog_desc)))

# pickle_file = "test5.pickle"
# gt = pickle.load( open( pickle_file, "rb" ) );
# # print(image)
# print(gt)
# print(hog_desc)
# # np.testing.assert_allclose(hog_desc,gt,rtol=1e-3)
# # print(inspect.stack()[0][3],' Passed')
