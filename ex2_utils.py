import numpy as np

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
	pass
	
def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):
	"""
	Detects edges using the Sobel method
	:param img: Input image
	:param thresh: The minimum threshold for the edge response
	:return: opencv solution, my implementation
	"""
	pass

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
