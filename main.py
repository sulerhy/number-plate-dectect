import cv2
import imutils
import numpy as np
import _utils
import processing
import sys
import CONST

np.set_printoptions(threshold=sys.maxsize)


def main():
    # load image
    img = _utils.read_img("9.jpg")
    img_result = img.copy()
    # get salient objects from image
    salientObjects, cars_no = processing.get_cars(img)
    # loop over the detections
    for i in range(0, min(cars_no, CONST.MAX_PROPOSED_OBJECTS)):
        # get the bounding box coordinates
        (startX, startY, endX, endY) = salientObjects[i].flatten()
        # 60% of the cropped area, for important information because plate_number always lay below car
        startY = int((startY + endY) * (1 - CONST.TARGETED_OBJECT))
        car_box = img[startY:endY, startX:endX]
        number_plate = processing.get_number_plate(car_box)
        if number_plate is not None:
            print("FOUND! Number_plate")
            # debugging
            # car_box_debugging = cv2.drawContours(car_box.copy(), number_plate, -1, (0, 255, 0), 3)
            # _utils.show_img("car_box", car_box_debugging)
            # end debugging
            cv2.drawContours(img_result, number_plate, -1, (0, 255, 0), thickness=3, offset=(startX, startY))
        else:
            print("NOT FOUND! Number_plate")
    _utils.show_img("result", img_result)


if __name__ == "__main__":
    # execute only if run as a script
    main()
