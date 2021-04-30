from ex2_utils import *
import matplotlib.pyplot as plt
import time


def main():
	studentID()
	#conv1Demo()
	#conv2Demo()
	derivDemo()
	#blurDemo()
	#edgeDemo()
	#houghDemo()

def studentID():
	print(get_studentID())
	
def conv1Demo():
	array = np.array([1, 2, 3, 4, 5])
	kernel = np.array([1, 2, 3, 4])
	print("numpy: " + str(np.convolve(array, kernel)))
	print("custom: " + str(conv1D(array, kernel))) 
	print((conv1D(array, kernel) == np.convolve(array, kernel, 'full')).all())
	
def conv2Demo():
	img_path = "data/beach.jpg"
	img = cv2.imread(img_path, 0)
	kernel = np.array([[0.0, -1.0, 0.0], 
						[-1.0, 4.0, -1.0],
						[0.0, -1.0, 0.0]])
	kernel = np.array([[0, -1, 0], 
						[-1, 4, -1],
						[0, -1, 0]])
	img_rst = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
	
	conv2D_output = conv2D(img, kernel)
	
	#print(conv2D_output.shape)
	
	print(img)
	print(kernel)
	#print((conv2D_output == img_rst).all())
	cv2.imshow("cv", img_rst)
	cv2.imshow("custom", conv2D_output)
	print(img_rst)
	print(conv2D_output)
	#conv2D_output = conv2D_output - np.min(conv2D_output)
	#conv2D_output = conv2D_output / (np.max(conv2D_output)/ 255)
	#conv2D_output = conv2D_output.astype(int)
	#print(np.max(conv2D_output))
	#cv2.imshow("custom", conv2D_output)
	#print(conv2D_output)
	cv2.waitKey()
	
def derivDemo():
	img_path = "data/coins.jpg"
	img = cv2.imread(img_path, 0)
	directions, magnitude, x_der, y_der = convDerivative(img)
	cv2.imshow("x_der", x_der)
	cv2.imshow("y_der", y_der)
	#cv2.imshow("magnitude", magnitude)
	cv2.imshow("directions", directions)
	cv2.waitKey()
	
def blurDemo():
	img_path = "data/pool_balls.jpeg"
	img = cv2.imread(img_path, 0)
	kernel = 1/16 * np.array([[1, 2, 1], 
							[2, 4, 2],
							[1, 2, 1]])
						
	opencv = blurImage2(img, kernel)
	cv2.imshow("opencv", opencv)
	cv2.imshow("orig", img)
	#print(cv2.getGaussianKernel(3, 1))
	cv2.waitKey()
	
def edgeDemo():
	img_path = "data/codeMonkey.jpeg"
	img = cv2.imread(img_path, 0)
	
	sobel_64 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	abs_64 = np.absolute(sobel_64)
	sobel_8u = np.uint8(abs_64)
	
	#edges = edgeDetectionSobel(img)
	
	#cv2.imshow("custom", edges)
	cv2.imshow("custom", sobel_8u)
	cv2.waitKey()
	
def houghDemo():
	pass

if __name__ == '__main__':
	main()
