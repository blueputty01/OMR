import cv2
# load the image and display it
image = cv2.imread("images/chinese.png")
image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# apply simple thresholding with a hardcoded threshold value
(T, threshInv) = cv2.threshold(blurred, 230, 255,
                               cv2.THRESH_BINARY_INV)
cv2.imshow("Simple Thresholding", threshInv)

# apply Otsu's automatic thresholding
(T, threshInv) = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Otsu Thresholding", threshInv)

# instead of manually specifying the threshold value, we can use
# adaptive thresholding to examine neighborhoods of pixels and
# adaptively threshold each neighborhood
thresh = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imshow("Mean Adaptive Thresholding", thresh)

cv2.waitKey(0)