# import the necessary packages
import math

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os


# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 2, 1: 3, 2: 0, 3: 3, 4: 1, 5: 1, 6: 1, 7: 3, 8: 3, 9: 2}


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

        section_contours = contours.sort_contours(section_contours, method="top-to-bottom")[0]
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


def grade_column(warped):
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
            # blank_image = cv2.putText(blank_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            question_contours.append(contour)

    question_contours = contours.sort_contours(question_contours, method="top-to-bottom")[0]

    i = 0
    for contour in question_contours:
        i += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        blank_image = cv2.putText(blank_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    blank_image = cv2.drawContours(blank_image, question_contours, -1, (0, 0, 255), 1)
    cv2.imshow("question contours", blank_image)

    correct = 0
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(question_contours), 4)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        sorted_bubbles = contours.sort_contours(question_contours[i:i + 4])[0]
        bubbled = None

        # loop over the sorted contours
        for (j, c) in enumerate(sorted_bubbles):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        # initialize the contour color and the index of the
        # *correct* answer
        color = (0, 0, 255)
        k = ANSWER_KEY[q]
        # check to see if the bubbled answer is correct
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        # draw the outline of the correct answer on the test
        print(k)
        # print(sorted_bubbles[k])
        cv2.drawContours(warped, [sorted_bubbles[k]], -1, color, 3)
    cv2.imshow("marked", warped)


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


    # load the image, convert it to grayscale, blur it slightly
    user_image = cv2.imread(args["image"]).copy()
    # TODO: used for debugging
    # user_image = cv2.resize(user_image, (0, 0), fx=0.5, fy=0.5)
    warped = get_section(7, 1.5, user_image)

    cv2.imshow("warped", warped)

    # isolate column
    columns = [[(7/8) / 7, (1 + 3/4) / 7]]
    for columnRange in columns:
        (original_height, original_width) = np.shape(warped)

        # print(columnRange[0])
        # print(columnRange[1])
        # print(original_width * columnRange[0])
        # print(original_width * columnRange[1])
        # print("h", original_height)
        # print("w", original_width)

        x1 = int(original_width * columnRange[0])
        x2 = int(original_width * columnRange[1])

        column_image = warped[0:original_height, x1:x2].copy()
        cv2.imshow("cropped?", column_image)
        grade_column(column_image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
