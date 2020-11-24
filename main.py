import cv2
import imutils
import numpy as np
import _utils
import processing
import sys
import CONST
from PlateDetection import PlateDetectionMain

np.set_printoptions(threshold=sys.maxsize)


def main():
    detected_flag = False
    # load image
    img = _utils.read_img("27.jpg")
    img_result = img.copy()
    # get salient objects from image
    salientObjects, cars_no = processing.get_cars(img)
    # loop over the detections
    for i in range(0, min(cars_no, CONST.MAX_PROPOSED_OBJECTS)):
        # get the bounding box coordinates
        (startX, startY, endX, endY) = salientObjects[i].flatten()
        # 60% of the cropped area, for important information because plate_number always lay below car
        startY = int(startY + (endY - startY) * (1 - CONST.TARGETED_OBJECT))
        car_box = img[startY:endY, startX:endX]
        number_plate = processing.get_number_plate(car_box)
        if number_plate is not None:
            detected_flag = True
            PlateDetectionMain.drawRedRectangleAroundPlate(img_result, number_plate, offset=(startX, startY))

    if detected_flag:
        print("------- number plate FOUNDED ----------")
    else:
        print("------- number plate NOT FOUNDED ----------")

    _utils.show_img("result", img_result)


if __name__ == "__main__":
    # execute only if run as a script
    main()
