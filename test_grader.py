# import the necessary packages
import math

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import cv2

# TODO: use auto-canny for better section extraction
# TODO: must work with blank responses

input_image = None
gray_input_image = None
key = []


def get_section(width, height):
    blurred = cv2.GaussianBlur(gray_input_image, (3, 3), 0)
    # TODO: smooth out auto canny instead of adaptive threshold?
    # https://stackoverflow.com/questions/24672414/adaptive-parameter-for-canny-edge
    # edged = auto_canny(blurred)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    contour_data, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                if check_contour_aspect_ratio(width, height, 100, 100, 0.1, approx):
                    section_contours.append(approx)

        section_contours = contours.sort_contours(section_contours, method="top-to-bottom")[0]
    # numpy.reshape: 4 arrays, each with 2 elements
    warped = four_point_transform(gray_input_image, section_contours[0].reshape(4, 2))
    return warped


def get_bubbles(img):
    img = cv2.GaussianBlur(img, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 5)
    # cv2.imshow("thresh", thresh)

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    all_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    height, width = thresh.shape
    blank_image = np.zeros((height, width, 3), np.uint8)

    all_image = cv2.drawContours(blank_image.copy(), all_contours, -1, (0, 0, 255), 1)
    # cv2.imshow("all contours", all_image)
    for contour in all_contours:
        if check_contour_aspect_ratio(112.952, 70.595, 40, 20, 0.5, contour):
            filtered_contours.append(contour)

    filtered_contours = contours.sort_contours(filtered_contours, method="top-to-bottom")[0]

    bubbles = cv2.drawContours(blank_image.copy(), filtered_contours, -1, (0, 0, 255), 1)
    # cv2.imshow("bubble contours", bubbles)

    return filtered_contours, thresh


def read_bubbles(threshed_image, question_contours):
    selections = []
    selection_contours = []
    # each question has 4 possible answers; loop over the
    # questions in batches of 4
    for (question_number, i) in enumerate(np.arange(0, len(question_contours), 4)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        sorted_bubbles = contours.sort_contours(question_contours[i:i + 4])[0]
        bubbled = -1

        # loop over the sorted contours
        for (option_number, c) in enumerate(sorted_bubbles):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(threshed_image.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(threshed_image, threshed_image, mask=mask)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            # TODO: this might cause problems with sizing
            # TODO: allow for no bubble
            if total > 1000:
                bubbled = option_number
        selections.append(bubbled)
        selection_contours.append(sorted_bubbles)
    return selections, selection_contours


def grade_column(output_image, bubble_contours, column_responses, key):
    for (i, answer) in enumerate(key):
        if answer == -1:
            continue
        selection = column_responses[i]
        # initialize the contour color and the index of the
        # *correct* answer
        color = (0, 0, 255)
        # check to see if the bubbled answer is correct
        if answer == selection:
            color = (0, 255, 0)
        # draw the outline of the correct answer on the test
        # print(sorted_bubbles[k])
        cv2.drawContours(output_image, [bubble_contours[i][answer]], -1, color, 3)
    return output_image


def check_contour_aspect_ratio(target_width, target_height, min_width, min_height, tolerance, contour):
    test_ratio = target_width / target_height
    # bounding box to derive the aspect ratio
    (x, y, width, height) = cv2.boundingRect(contour)
    ratio = width / float(height)

    # todo: size the image so that these values work no matter input size
    if width >= min_width and height >= min_height:
        if math.isclose(ratio, test_ratio, rel_tol=tolerance):
            return True
    return False


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


def crop_border(image, border):
    (h, w) = np.shape(image)
    crop = image[border:h - border, border: w - border]
    return crop


def main():
    # TODO: read in all images
    global input_image
    global gray_input_image
    input_image = cv2.imread("images/IMG_2458.jpg")
    resize_with_aspect_ratio(input_image, 3024, 4032)
    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    constant_key = [
        [
            [1, 2, 3, 0, 1],
            [3, 1, 1, 1, 1],
            [3, 1, 0, 2, 3],
            [0, 0, 3, 1, 1, 3, 2, 2, 3, 0],
            [3, 2, 1, 3, 0, 2, 3]
        ]
    ]
    grading = False
    sections = [
        {
            'dimensions': [5 + 5 / 8, 2],
            'top_offset': 0.5,
            'columns': 5
        },
        {
            'dimensions': [5 + 5 / 8, 1.75],
            'top_offset': 0.2,
            'columns': 4
        }
    ]
    global key
    if not grading:
        key = []

    for i, sec_info in enumerate(sections):
        if not grading:
            key.append([])

        dimensions = sec_info['dimensions']
        section_image = get_section(dimensions[0], dimensions[1])
        section_crop = crop_border(section_image, 10)

        values = []
        for j in range(sec_info['columns']):
            (original_height, original_width) = np.shape(section_crop)

            offset = j * 1.1 + 7 / 16

            x1 = int(original_width * offset / dimensions[0])
            x2 = int(original_width * (offset + 3 / 4) / dimensions[0])

            y1 = int(original_height * sec_info['top_offset'] / dimensions[1])

            column_image = section_crop[y1:original_height, x1:x2].copy()
            # TODO: this is a bit circuitous?: going to and from gray to rgb
            # cv2.imshow("column " + str(j), column_image)
            bubbles, thresh = get_bubbles(column_image)
            column_selections, selected_bubbles = read_bubbles(thresh, bubbles)

            if grading:
                color_column = cv2.cvtColor(column_image, cv2.COLOR_GRAY2BGR)
                graded_column = grade_column(color_column, selected_bubbles, column_selections, constant_key[0][i])
                cv2.imshow("graded", graded_column)
            else:
                key[i].append(column_selections)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
