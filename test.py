import cv2
import imutils
import _utils
import processing
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
car_box = _utils.read_img("test_box.png")
number_plate = processing.get_number_plate(car_box)

