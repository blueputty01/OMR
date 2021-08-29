# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    cannied = cv2.Canny(image, lower, upper)
    # return the edged image
    return cannied


def get_paper_contours(contour_data):
    paper_contours = None
    # ensure that at least one contour was found
    if len(contour_data) > 0:
        # sort the contours according to their size in
        # descending order
        contour_data = sorted(contour_data, key=cv2.contourArea, reverse=True)
        billy = contour_data[0]
        print(billy)
        perimeter = cv2.arcLength(billy, True)
        print(perimeter)
        approx = cv2.approxPolyDP(billy, 0.1 * perimeter, True)
        print(f'approx: {approx}')
        bob = np.zeros((4032, 3024, 3), np.uint8)
        for i in range(30):
            bob = cv2.polylines(bob, [contour_data[i]], False, (0, 255, 0), 10)

        hi = resize_with_aspect_ratio(bob, height=700)
        cv2.imshow("image", hi)
        # loop over the sorted contours
        for c in contour_data:
            # approximate the contour
            # read up on this step!
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            #print(f'shape: {len(approx)}')
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                paper_contours = approx
                break
    #print(paper_contours)
    return paper_contours


def get_paper(passed_image):
    # make the picture smaller for development purposes
    # otherwise, the window takes up the whole screen lol
    gray = cv2.cvtColor(passed_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = auto_canny(blurred)

    new = resize_with_aspect_ratio(edged, height=700)
    cv2.imshow("edged", new)

    # find edges
    # ret, thresh = cv2.threshold(edged, 127, 255, 0)
    cntrs, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(f'contours found: {len(cntrs)}')
    cv2.drawContours(passed_image, cntrs, -1, (0, 0, 255), 3)

    doc_contours = get_paper_contours(cntrs)

    # TODO: remove; for debugging contours:
    passed_image = cv2.polylines(passed_image, [doc_contours], True, (0, 255, 0), 3)

    new = resize_with_aspect_ratio(passed_image, height=700)
    cv2.imshow("Edges", new)

    os.chdir("./")
    cv2.imwrite("temp.png", passed_image)


    # numpy.reshape: 4 arrays, each with 2 elements
    warped = four_point_transform(gray, doc_contours.reshape(4, 2))
    return warped


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# construct the argument parse and parse the arguments
# noinspection DuplicatedCode
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# load the image, convert it to grayscale, blur it slightly
user_image = cv2.imread(args["image"]).copy()
# TODO: used for debugging
# user_image = cv2.resize(user_image, (0, 0), fx=0.5, fy=0.5)
warped = get_paper(user_image)

# warped = cv2.GaussianBlur(warped, (3, 3), 0)
cv2.imshow("warped", warped)
# fifth parameter must be odd
# thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 40)
thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
bubble_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

# #print(bubble_contours)
# questionCnts = []
# # loop over the contours
# for c in bubble_contours:
#     # compute the bounding box of the contour, then use the
#     # bounding box to derive the aspect ratio
#     (x, y, w, h) = cv2.boundingRect(c)
#     ar = w / float(h)
#     # in order to label the contour as a question, region
#     # should be sufficiently wide, sufficiently tall, and
#     # have an aspect ratio approximately equal to 1
#     if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
#         questionCnts.append(c)

height, width = thresh.shape
blank_image = np.zeros((height, width, 3), np.uint8)
cv2.drawContours(blank_image, bubble_contours, -1, (0, 0, 255), 1)

cv2.imshow("threshold", thresh)
cv2.imshow("contours", blank_image)
cv2.waitKey(0)
