# import the necessary packages
import math

from imutils.perspective import four_point_transform
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


def get_section(width, height, passed_image):

    # find edges
    # ret, thresh = cv2.threshold(edged, 127, 255, 0)

    gray = cv2.cvtColor(passed_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = auto_canny(blurred)
    # TODO: smooth out auto canny instead of adaptive threshold?
    # https://stackoverflow.com/questions/24672414/adaptive-parameter-for-canny-edge
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    contour_data, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(passed_image, contour_data, -1, (0, 0, 255), 3)

    os.chdir("./")
    cv2.imwrite("thresh.png", thresh)
    cv2.imwrite("edged.png", edged)

    # resized = resize_with_aspect_ratio(passed_image, height=700)
    # cv2.imshow("all contours", resized)
    # ensure that at least one contour was found
    test_ratio = width / height
    section_contours = []
    if len(contour_data) > 0:
        # sort the contours according to their size in
        # descending order
        contour_data = sorted(contour_data, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in contour_data:
            # approximate the contour
            # read up on this step!
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                # compute the bounding box of the contour, then use the
                # bounding box to derive the aspect ratio
                # todo: size the image so that these values work no matter input size
                if valid_contour(width, height, 100, 100, 0.1, approx):
                    section_contours.append(approx)

        def contour_height(ele):
            bounding_rect = cv2.boundingRect(ele)
            return bounding_rect[1]

        section_contours = sorted(section_contours, key=contour_height, reverse=True)

    # numpy.reshape: 4 arrays, each with 2 elements
    warped = four_point_transform(gray, section_contours[0].reshape(4, 2))
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


def get_bubbles(warped):
    print('HI')
    # fifth parameter must be odd
    # TODO: use text recognition?
    warped = cv2.GaussianBlur(warped, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 15)
    cv2.imshow("thresh", thresh)

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    all_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    question_contours = []

    height, width = thresh.shape
    blank_image = np.zeros((height, width, 3), np.uint8)

    all_image = cv2.drawContours(blank_image.copy(), all_contours, -1, (0, 0, 255), 1)
    cv2.imshow("all contours", all_image)

    i = 0
    for contour in all_contours:
        i += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        if valid_contour(1/8, 1/16, 10, 10, 0.4, contour):
            print("valid", i)
            blank_image = cv2.putText(blank_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            question_contours.append(contour)
        # else:
        #     # print("NO")

    blank_image = cv2.drawContours(blank_image, question_contours, -1, (0, 0, 255), 1)
    cv2.imshow("question contours", blank_image)


def valid_contour(target_width, target_height, min_width, min_height, tolerance, contour):
    test_ratio = target_width / target_height
    # bounding box to derive the aspect ratio
    (x, y, width, height) = cv2.boundingRect(contour)
    ratio = width / float(height)

    # todo: size the image so that these values work no matter input size
    if width >= min_width and height >= min_height:
        if math.isclose(ratio, test_ratio, rel_tol=tolerance):
            return True
    return False


def main():
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
    warped = get_section(7, 1.5, user_image)

    cv2.imshow("warped", warped)

    get_bubbles(warped)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
