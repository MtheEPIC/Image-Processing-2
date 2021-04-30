import numpy as np
import cv2

"""
The input of all the functions will be grayscale images
There is no input validation
"""

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
	#return cv2.filter2D(inImage, -1, kernel2, borderType=cv2.BORDER_REPLICATE)
	tmp = kernel2.copy()
	print(kernel2)
	tmp[0, :] = kernel2[2, :]
	tmp[2, :] = kernel2[0, :]
	kernel2 = tmp
	tmp = kernel2.copy()
	tmp[:, 0] = kernel2[:, 2]
	tmp[:, 2] = kernel2[:, 0]
	kernel2 = tmp
	print(kernel2)
	
	outImage = np.zeros((inImage.shape))
	inImage = np.pad(inImage, (kernel2.shape[0]-1, kernel2.shape[1]-1), 'constant', constant_values=(0))
	tmpImage = np.zeros((inImage.shape))
	
	for i in range(kernel2.shape[0]-1, kernel2.shape[0]-1 + outImage.shape[0]): #not padded area
		for j in range(kernel2.shape[1]-1, kernel2.shape[1]-1 + outImage.shape[1]): #not padded area
			#calc val
			tmpImage[i, j] = np.multiply(inImage[i:i+kernel2.shape[0], j:j+kernel2.shape[1]], kernel2).sum()
			"""
			val = 0
			for m in range(-kernel2.shape[0]//2, kernel2.shape[0]//2 +1):
				for n in range(-kernel2.shape[1]//2, kernel2.shape[1]//2 +1):
					val += kernel2[m+kernel2.shape[0]//2, n+kernel2.shape[1]//2]*inImage[i-m, j-n]
			tmpImage[i, j] = val
			"""
	
	outImage = tmpImage[kernel2.shape[0]-1: kernel2.shape[0]-1 + outImage.shape[0], kernel2.shape[1]-1: kernel2.shape[1]-1 + outImage.shape[1]]
	return outImage
	
	#outImage = np.zeros((inImage.shape[0]-kernel2.shape[0]+1, inImage.shape[1]-kernel2.shape[1]+1))
	outImage = np.zeros((inImage.shape))
	inImage = np.pad(inImage, (1, 1), 'constant', constant_values=(0))
	#outImage = np.pad(outImage, (1, 1), 'constant', constant_values=(0))
	print("ddf")
	for i in range(outImage.shape[0]):
		for j in range(outImage.shape[1]):
			outImage[i, j] = np.multiply(inImage[i:i+kernel2.shape[0], j:j+kernel2.shape[1]], kernel2).sum()
	return outImage.astype(int)

def convDerivative(inImage: np.ndarray) ->(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
	"""
	Calculate gradient of an image
	:param inImage: Grayscale iamge
	:return: (directions, magnitude, x_der, y_der)
	"""
	kernel = np.array([-1, 0, 1])
	x_der = cv2.filter2D(inImage, -1, np.transpose([kernel]), borderType=cv2.BORDER_REPLICATE)
	y_der = cv2.filter2D(inImage, -1, kernel, borderType=cv2.BORDER_REPLICATE)
	
	kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	kernel_x = np.flipud(np.fliplr(kernel_x))
	kernel_y = np.flipud(np.fliplr(kernel_y))
	x_der = cv2.filter2D(inImage, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
	y_der = cv2.filter2D(inImage, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
	
	print(x_der)
	print(y_der)
	
	x_der[x_der == 0] = 1
	print(x_der)
	
	magnitude = np.sqrt(x_der**2 + y_der**2)
	
	directions = np.arctan(y_der/ x_der)
	
	return directions, magnitude, x_der, y_der

def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray)-> np.ndarray:
	"""
	Blur an image using a Gaussian kernel
	:param inImage: Input image2
	:param kernelSize: Kernel size
	:return: The Blurred image
	"""
	pass
	
def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray)-> np.ndarray:
	"""
	Blur an image using a Gaussian kernel using OpenCV built-in functions
	:param inImage: Input image
	:param kernelSize: Kernel size
	:return: The Blurred image
	"""
	kernel = cv2.getGaussianKernel(kernel_size.shape[0], 1)
	return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
	
def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):
	"""
	Detects edges using the Sobel method
	:param img: Input image
	:param thresh: The minimum threshold for the edge response
	:return: opencv solution, my implementation
	"""
	Gx = np.dot(img, [-1, 0, 1])
	return img

def edgeDetectionZeroCrossingSimple(img: np.ndarray)-> (np.ndarray):
	"""
	Detecting edges using the "ZeroCrossing" method
	:param img: Input image
	:return: Edge matrix
	"""
	pass
	
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
	pass
	
def houghCircle(img: np.ndarray, min_radius: float, max_radius: float)-> list:
	"""
	Find Circles in an image using a Hough Transform algorithm extension
	:param I: Input image
	:param minRadius: Minimum circle radius
	:param maxRadius: Maximum circle radius
	:return: A list containing the detected circles,[(x,y,radius),(x,y,radius),...]
	"""
	pass
