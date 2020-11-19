import cv2
import imutils
import numpy as np
import _utils


def main():
    print("Start Processing...")
    img = _utils.read_img("192.168.1.64_01_20200730111229020_MOTION_DETECTION.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 13, 60, 60)
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in contours:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    # mask = np.zeros(gray.shape, np.uint8)
    # new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    # new_image = cv2.bitwise_and(img, img, mask=mask)

    # debugging
    _utils.show_img(img)


if __name__ == "__main__":
    # execute only if run as a script
    main()
