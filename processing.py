import cv2
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


def get_number_plate(car_bbox):
    """
    Get number plate in each proposed car object (car_bbox)
    :param car_bbox: proposed car object
    """
    lic_plate = PlateDetectionInObject.get_number_plate(car_bbox)
    return lic_plate
