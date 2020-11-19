import cv2
import numpy as np


def read_img(img_name):
    img = cv2.imread("resources/input_images/" + img_name, cv2.IMREAD_COLOR)
    return img


def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
