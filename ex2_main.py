from ex2_utils import *
import matplotlib.pyplot as plt
import time


def main():
	studentID()
	#conv1Demo()
	#conv2Demo()
	#derivDemo()
	#edgeDemo()
	#blurDemo()
	houghDemo()

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
	img = load_img(img_path)
	kernel = np.array([[0, -1, 0], 
						[-1, 4, -1],
						[0, -1, 0]])
	arr =np.zeros(shape=(5,5))
	for i in range(5):
		for j in range(5):
			arr[i][j]=(j+1)+i*5
	img = arr
	conv2D_cv2 = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
	conv2D_output = conv2D(img, kernel)
	
	cv2.imshow("cv", conv2D_cv2)
	cv2.imshow("custom", conv2D_output)
	print("cv2: \n" + str(conv2D_cv2))
	print("custom: \n" + str(conv2D_output))
	print((conv2D_output.astype('float16') == conv2D_cv2.astype('float16')).all())
	cv2.waitKey()
	
def derivDemo():
	img_path = "data/coins.jpg"
	img = cv2.imread(img_path, 0)
	directions, magnitude, x_der, y_der = convDerivative(img)
	cv2.imshow("x_der", x_der)
	cv2.imshow("y_der", y_der)
	cv2.imshow("magnitude", magnitude)
	cv2.imshow("directions", directions)
	cv2.waitKey()
	
def blurDemo():
	img_path = "data/pool_balls.jpeg"
	img = load_img(img_path)
	kernel = 1/16 * np.array([[1, 2, 1], 
							[2, 4, 2],
							[1, 2, 1]])
						
	custom = blurImage1(img, kernel)
	opencv = blurImage2(img, kernel)
	cv2.imshow("custom", custom)
	cv2.imshow("opencv", opencv)
	cv2.imshow("orig", img)
	cv2.waitKey()
	
def edgeDemo():
	img_path = "data/codeMonkey.jpeg"
	img = cv2.imread(img_path, 0)
	img_scaled = load_img(img_path)
	
	edges_sobel_opencv, edges_sobel = edgeDetectionSobel(img, 20)
	edges_canny = edgeDetectionCanny(img, 20, 50)
	edges_zerocross = edgeDetectionZeroCrossingSimple(img_scaled)
	
	cv2.imshow("ZeroCrossingSimple", edges_zerocross)
	cv2.imshow("edgeDetectionCanny", edges_canny)
	cv2.imshow("Sobel mine", edges_sobel)
	cv2.imshow("Sobel opencv", edges_sobel_opencv)
	cv2.waitKey()
	
def houghDemo():
	img_path = "data/coins.jpg"
	img = cv2.imread(img_path, 0)
	
	circles = houghCircle(img, 5, 10)
	
	plt.imshow(img)
	for circle in circles:
		cc = plt.Circle((circle[0], circle[1]), circle[2]) 
		plt.gca().add_artist(cc)
	plt.show()

if __name__ == '__main__':
	main()
