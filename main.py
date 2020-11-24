import cv2
import imutils
import numpy as np
import _utils
import processing
import sys
import CONST
from PlateDetection import PlateDetectionInObject
import glob

np.set_printoptions(threshold=sys.maxsize)
root_folder = "/Users/sulerhy/Desktop/number-plate-dectect/"


def main():
    # get list input files
    list_images = glob.glob(root_folder + "resources/input_images/" + "*.jpg")
    for img_name in list_images:
        # read image from input folder
        img = cv2.imread(img_name)
        result_img = print_bbox(img)
        # debugging
        # _utils.show_img("result", result_img)
        # write image to output folder
        output_name = img_name.replace("input_images", "output_images")
        cv2.imwrite(output_name, result_img)


def print_bbox(img):
    """
    print bbox of number plate on image
    :param img:
    :return: img_result
    """
    detected_flag = False
    # load image
    img_result = img.copy()
    # get salient objects from image
    salientObjects, cars_no = processing.get_cars(img)
    # loop over the detections
    print("Step 2: take number plate on each car")
    for i in range(0, min(cars_no, CONST.MAX_PROPOSED_OBJECTS)):
        # get the bounding box coordinates
        (startX, startY, endX, endY) = salientObjects[i].flatten()
        # 60% of the cropped area, for important information because plate_number always lay below car
        startY = int(startY + (endY - startY) * (1 - CONST.TARGETED_OBJECT))
        car_box = img[startY:endY, startX:endX]
        number_plate = processing.get_number_plate(car_box)
        if number_plate is not None:
            detected_flag = True
            PlateDetectionInObject.drawRedRectangleAroundPlate(img_result, number_plate, offset=(startX, startY))
    if detected_flag:
        print("------- number plate FOUNDED ----------")
    else:
        print("------- number plate NOT FOUNDED ----------")
    return img_result


if __name__ == "__main__":
    # execute only if run as a script
    main()
