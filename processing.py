import cv2
import imutils
import CONST
import _utils
from PlateDetection import PlateDetectionInObject

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
    print("Step1: Start finding vehicles...")
    # OpenCV's objectness saliency detector
    saliency = cv2.saliency.ObjectnessBING_create()
    # load trained model of OpenCV
    saliency.setTrainingPath("ObjectnessTrainedModel")
    # compute the bounding box predictions used to indicate saliency
    (success, saliencyMaps) = saliency.computeSaliency(image)
    cars_no = saliencyMaps.shape[0]
    print("----Number of the detected objects:" + str(cars_no))
    return saliencyMaps, cars_no


#########################　METHOD 1: Googleで参考したメソッド　######################################
def get_number_plate(car_bbox):
    """
    Get number plate in each proposed car object (car_bbox)
    :param car_bbox: proposed car object
    """
    lic_plate = PlateDetectionInObject.get_number_plate(car_bbox)
    return lic_plate


#########################　METHOD 2: 古いメソッド(使用しない)　######################################
# def get_number_plate(car_bbox):
#     """
#     get number plate of the cars after get_cars
#     :param car_bbox: input proposed cars (bounding box of the car)
#     """
#     print("Start getting car number plate...")
#     # _utils.show_img("car_bbox", car_bbox)
#     # gray scale
#     gray = cv2.cvtColor(car_bbox, cv2.COLOR_BGR2GRAY)  # convert to grey scale
#     gray = cv2.bilateralFilter(gray, 5, 35, 35)
#     # get edged
#     edged = cv2.Canny(gray, 30, 200)
#     _utils.show_img("edged", edged)
#
#     # get contours
#     contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(contours)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:CONST.MAX_TAKEN_CONTOUR]
#     # debugging draw all contours
#     # car_box_debugging = cv2.drawContours(car_bbox.copy(), contours, -1, CONST.CONTOUR_COLOR, CONST.CONTOUR_SIZE)
#     # _utils.show_img("car_box_debugging", car_box_debugging)
#     rectangle_detected = None
#     for c in contours:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, CONST.APRROX_POLY_DP * peri, True)
#         if len(approx) == 4:
#             rectangle_detected = approx
#             break
#
#     if rectangle_detected is None:
#         return None
#     else:
#         return [rectangle_detected]
