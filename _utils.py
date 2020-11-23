import cv2


def read_img(img_name):
    img = cv2.imread("resources/input_images/" + img_name, cv2.IMREAD_COLOR)
    return img


def show_img(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
