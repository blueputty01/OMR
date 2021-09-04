# import the necessary packages
import math

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import cv2

# TODO: use auto-canny for better section extraction

input_image = None
gray_input_image = None


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
    cv2.imshow("thresh", thresh)

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    all_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    height, width = thresh.shape
    blank_image = np.zeros((height, width, 3), np.uint8)

    all_image = cv2.drawContours(blank_image.copy(), all_contours, -1, (0, 0, 255), 1)
    cv2.imshow("all contours", all_image)
    for contour in all_contours:
        if check_contour_aspect_ratio(112.952, 70.595, 9, 9, 0.5, contour):
            filtered_contours.append(contour)

    filtered_contours = contours.sort_contours(filtered_contours, method="top-to-bottom")[0]

    bubbles = cv2.drawContours(blank_image.copy(), filtered_contours, -1, (0, 0, 255), 1)
    cv2.imshow("bubble contours", bubbles)

    return filtered_contours

def grade_column():
    return

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


def main():
    def crop_border(image, border):
        (h, w) = np.shape(image)
        crop = section_image[border:h - border, border: w - border]
        return crop

    # load the image, convert it to grayscale, blur it slightly
    # TODO: read in all images
    global input_image
    global gray_input_image
    input_image = cv2.imread("images/IMG_2458.jpg")
    resize_with_aspect_ratio(input_image, 3024, 4032)
    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    key = [
        [1, 2, 3, 0, 1, 3, 1, 1, 1, 1, 3, 1, 0, 2, 3, 0, 0, 3, 1, 1, 3, 2, 2, 3, 0, 3, 2, 1, 3, 0, 2, 3]
    ]
    grading = True
    sections = [
        {
            'dimensions': [5 + 5 / 8, 2],
            'top_offset': 0.5,
            'columns': 5
        }
    ]
    for section in sections:
        dimensions = section['dimensions']
        section_image = get_section(dimensions[0], dimensions[1])
        section_crop = crop_border(section_image, 10)

        bubble_contours = []
        for i in range(section['columns']):
            (original_height, original_width) = np.shape(section)

            offset = i * 1.1 + 7 / 16

            x1 = int(original_width * offset / dimensions[0])
            x2 = int(original_width * (offset + 3 / 4) / dimensions[0])

            y1 = int(original_height * section['top_offset'] / dimensions[1])

            column_image = section[y1:original_height, x1:x2].copy()
            cv2.imshow("column " + str(i), column_image)
            bubble_contours.append(get_bubbles(column_image))

            if grading:
                graded_column = grade_column()

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
