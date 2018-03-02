from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def edgeDetection(image):
	image = imutils.resize(image, height = 500)
	 
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)
	 
	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return edged


def findContour(edged):

	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	 
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	 
		if len(approx) == 4:
			screenCnt = approx
			break
	 
	# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return screenCnt


def scan(screenCnt, image):
	ratio = image.shape[0] / 500.0

	warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
	 
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255

	# kernel = np.ones((1,5), np.uint8)  # note this is a HORIZONTAL kernel
	# kernel = np.array([(0,1,0),(1,1,1),(0,1,0)])
	# e_im = cv2.dilate(warped, kernel, iterations=1)
	# e_im = cv2.erode(e_im, kernel, iterations=2) 

	# cv2.imshow("Original", imutils.resize(orig, height = 650))
	# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	# cv2.imshow("Scanne", imutils.resize(e_im, height = 650))
	# cv2.waitKey(0)

	return warped

def main():
	image = cv2.imread("pic2.jpg")
	edged = edgeDetection(image)
	screenCnt = findContour(edged)
	scannedImage = scan(screenCnt,image)
	cv2.imshow("Scanned", imutils.resize(scannedImage, height = 650))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()