import cv2
import imutils
import pytesseract
import _utils
import numpy as np
import CONST

"""
Processing to get number plate of parking lot
@author: Pham Duc Hoang
@created: 2020/11/20
"""


def get_cars(image):
    """
    Objectness Detection by Saliency detection method
    :param image: The input image
    """
    print("Start finding vehicles...")
    # OpenCV's objectness saliency detector
    saliency = cv2.saliency.ObjectnessBING_create()
    # load trained model of OpenCV
    saliency.setTrainingPath("ObjectnessTrainedModel")
    # compute the bounding box predictions used to indicate saliency
    (success, saliencyMaps) = saliency.computeSaliency(image)
    cars_no = saliencyMaps.shape[0]
    print("----Number of the detected objects:" + str(cars_no))
    return saliencyMaps, cars_no


def get_number_plate(car_bbox):
    """
    get number plate of the cars after get_cars
    :param car_bbox: input proposed cars (bounding box of the car)
    """
    print("Start getting car number plate...")
    # gray scale
    gray = cv2.cvtColor(car_bbox, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 18, 18)
    # get edged
    edged = cv2.Canny(gray, 30, 200)
    _utils.show_img("edged", edged)
    # get contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:CONST.MAX_TAKEN_CONTOUR]
    _utils.show_img("car_bbox", car_bbox)
    # debugging draw all contours
    car_box_debugging = cv2.drawContours(car_bbox.copy(), contours, -1, CONST.CONTOUR_COLOR, CONST.CONTOUR_SIZE)
    _utils.show_img("car_box_debugging", car_box_debugging)
    rectangle_detected = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, CONST.APRROX_POLY_DP * peri, True)
        if len(approx) == 4:
            rectangle_detected = approx
            break

    # text = get_number_plate_text([rectangle_detected], car_bbox, gray)
    # print(text)

    if rectangle_detected is None:
        return None
    else:
        return [rectangle_detected]


def get_number_plate_text(rectangle, origin_img, origin_gray):
    # Masking the part other than the number plate
    mask = np.zeros(origin_gray.shape, np.uint8)
    cv2.drawContours(mask, rectangle, 0, 255, -1, )
    number_plate_alone = cv2.bitwise_and(origin_img, origin_img, mask=mask)
    _utils.show_img("number_plate_alone", number_plate_alone)
    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = origin_gray[topx:bottomx + 1, topy:bottomy + 1]
    # Read the number plate
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected Number is:", text)
    return text

