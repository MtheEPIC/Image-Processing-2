import numpy as np
import matplotlib.image as mpimg
import cv2

"""
The input of all the functions will be grayscale images
There is no input validation
"""

RGB_TO_GRAY = [0.2989, 0.5870, 0.1140]


def load_img(filename: str)-> np.ndarray:
	"""
	retrun img in grayscale from the given path
	:param filename: path as string
	:return: loaded grayscale image
	"""
	img = mpimg.imread(filename) / 255
	return np.dot(img, RGB_TO_GRAY)

def get_studentID()-> int:
	"""
	returns the student's ID
	:return: ID
	"""
	return 212245757
	
def conv1D(inSignal: np.ndarray, kernel1: np.ndarray)-> np.ndarray:
	"""
	Convolve a 1-D array with a given kernel
	:param inSignal: 1-D array
	:param kernel1: 1-D array as a kernel
	:return: The convolved array
	"""
	kernel1 = kernel1[::-1]
	outSignal = np.zeros([inSignal.shape[0] + kernel1.shape[0] - 1])
	# add edges to inSignal
	inSignal = np.pad(inSignal, (kernel1.shape[0] - 1, kernel1.shape[0] - 1), 'constant', constant_values=(0))
	
	# move the flipped kernel and multiply the values of the kernel and inSignal in the given position
	for i in range(inSignal.shape[0] - kernel1.shape[0] + 1):
		outSignal[i] = np.dot(inSignal[i:i+kernel1.shape[0]], kernel1)

	return outSignal.astype(int)
	
def conv2D(inImage: np.ndarray, kernel2: np.ndarray)-> np.ndarray:
	"""
	Convolve a 2-D array with a given kernel
	:param inImage: 2D image
	:param kernel2: A kernel1
	:return: The convolved image
	"""
	kernel2 = np.flipud(np.fliplr(kernel2))
	k_h = kernel2.shape[0]
	k_w = kernel2.shape[1]
	offset_h = k_h//2
	offset_w = k_w//2
	outImage = np.zeros((inImage.shape))
	
	# pad same as cv2.BORDER_REPLICATE
	inImage = np.pad(inImage, (k_h//2+1, k_w//2+1), 'edge')
	tmpImage = np.zeros((inImage.shape))
	
	# go over not padded area and calc val in cell
	for i in range(k_h//2, k_h//2 + outImage.shape[0]+1):
		for j in range(k_w//2, k_w//2 + outImage.shape[1]+1):
			tmpImage[i, j] = np.multiply(inImage[i-offset_h:i+offset_h+1, j-offset_w:j+offset_w+1], kernel2).sum()
	
	# trim padding
	outImage = tmpImage[kernel2.shape[0]-1: kernel2.shape[0]-1 + outImage.shape[0], kernel2.shape[1]-1: kernel2.shape[1]-1 + outImage.shape[1]]
	return outImage

def convDerivative(inImage: np.ndarray) ->(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
	"""
	Calculate gradient of an image
	:param inImage: Grayscale iamge
	:return: (directions, magnitude, x_der, y_der)
	"""
	# kernels to get derivatives
	kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	kernel_x = np.flipud(np.fliplr(kernel_x))
	kernel_y = np.flipud(np.fliplr(kernel_y))
	# get derivative
	x_der = cv2.filter2D(inImage, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
	y_der = cv2.filter2D(inImage, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
	# calc magnitude
	magnitude = np.sqrt(x_der**2 + y_der**2).astype('uint8')
	# calc directions
	directions = np.arctan(y_der/ x_der)
	
	return directions, magnitude, x_der, y_der

def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray)-> np.ndarray:
	"""
	Blur an image using a Gaussian kernel
	:param inImage: Input image2
	:param kernelSize: Kernel size
	:return: The Blurred image
	"""
	#kernel = PascalTriangle(kernel_size.shape[0])
	#tmp = np.dot(np.ones((3,3)), kernel)
	#print(np.transpose([kernel]))
	kernel = kernel_size
	return conv2D(in_image, kernel)

def PascalTriangle(depth: int)-> np.ndarray:
	"""
	Get last layer in pascal triangle for the given depth
	:param depth: number of the layer to get
	:return: desired layer values
	"""
	layer = [1]
	pad = [0]
	for x in range(depth-1):
		layer=[left+right for left,right in zip(layer+pad, pad+layer)]
	return layer

def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray)-> np.ndarray:
	"""
	Blur an image using a Gaussian kernel using OpenCV built-in functions
	:param inImage: Input image
	:param kernelSize: Kernel size
	:return: The Blurred image
	"""
	kernel = cv2.getGaussianKernel(kernel_size.shape[0], 1)
	return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
	
def edgeDetectionSobel(img: np.ndarray, thresh: float = .7)-> (np.ndarray, np.ndarray):
	"""
	Detects edges using the Sobel method
	:param img: Input image
	:param thresh: The minimum threshold for the edge response
	:return: opencv solution, my implementation
	"""
	img = cv2.blur(img, (3, 3)) 
	# get x, y derivatives
	x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
	y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
	x_abs = cv2.convertScaleAbs(x_grad)
	y_abs = cv2.convertScaleAbs(y_grad)
	# pass throgh the threshold
	x_abs[x_abs < thresh] = 0
	y_abs[y_abs < thresh] = 0
	# reassemble into one matrix
	opencv = cv2.addWeighted(x_abs, 0.5, y_abs, 0.5, 0)
	
	x_grad, y_grad = convDerivative(img)[2:]
	x_abs = np.abs(x_grad)
	y_abs = np.abs(y_grad)
	# pass throgh the threshold
	x_abs[x_abs < thresh] = 0
	x_abs[x_abs >= thresh] = 255
	y_abs[y_abs < thresh] = 0
	y_abs[y_abs >= thresh] = 255
	# reassemble into one matrix
	sobel = x_abs + y_abs
	
	return opencv, sobel

def edgeDetectionZeroCrossingSimple(img: np.ndarray)-> (np.ndarray):
	"""
	Detecting edges using the "ZeroCrossing" method
	:param img: Input image
	:return: Edge matrix
	"""
	img_blured = blurImage2(img, np.arange(5))
	
	kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	edges = cv2.filter2D(img_blured, -1, kernel, borderType=cv2.BORDER_REPLICATE)
	
	return edges
	
def edgeDetectionZeroCrossingLOG(img: np.ndarray)-> (np.ndarray):
	"""
	Detecting edges using the "ZeroCrossingLOG" method
	:param img: Input image
	:return: :return: Edge matrix
	"""
	pass
	
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)-> (np.ndarray, np.ndarray):
	"""
	Detecting edges usint "Canny Edge" method
	:param img: Input image
	:param thrs_1: T1
	:param thrs_2: T2
	:return: opencv solution, my implementation
	"""
	img_blured = blurImage2(img, np.arange(5))
	return cv2.Canny(img, thrs_1, thrs_2);
	
def houghCircle(img: np.ndarray, min_radius: float, max_radius: float)-> list:
	"""
	Find Circles in an image using a Hough Transform algorithm extension
	:param I: Input image
	:param minRadius: Minimum circle radius
	:param maxRadius: Maximum circle radius
	:return: A list containing the detected circles,[(x,y,radius),(x,y,radius),...]
	"""
	A = np.zeros((img.shape[0], img.shape[1], 360))
	img_edges = edgeDetectionCanny(img, 30, 100)
	
	img_edges = img_edges / np.max(img_edges)
	edges = np.argwhere(img_edges!=0)
	h = edges.shape[0]
	w = edges.shape[1]
	
	r = min_radius
	for x in range(h):
		for y in range(w):
			for t in range(360):
				for r in np.linspace(min_radius, maxRadius, 4):
					a = x - (r * np.cos(t * np.pi / 180)).astype('uint8')
					b = y - (r * np.sin(t * np.pi / 180)).astype('uint8')
					
					if a >= w or a < 0:
						continue
					if b >= h or b < 0:
						continue
					
					A[a, b, r] += 1
				
	
	return np.unique(A)
