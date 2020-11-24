# PlateDetectionInObject.py

import cv2
import numpy as np
import os
import operator
import _utils


from PlateDetection import DetectChars, DetectPlates, PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)


###################################################################################################
def get_number_plate(imgOriginalScene):
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training

    if not blnKNNTrainingSuccessful:  # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return  # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates

    # _utils.show_img("imgOriginalScene", imgOriginalScene)  # show scene image

    if len(listOfPossiblePlates) == 0:  # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
        return None
    else:
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        # cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
        # cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:  # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return None
        # end if

        # _utils.show_img("imgPlate", licPlate.imgPlate)

        return licPlate


# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate, offset):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect
    # number_plate coordinate + offset coordinate of car_bbox
    point_0 = tuple(p2fRectPoints[0])
    point_1 = tuple(p2fRectPoints[1])
    point_2 = tuple(p2fRectPoints[2])
    point_3 = tuple(p2fRectPoints[3])
    point_0 = tuple(map(operator.add, point_0, offset))
    point_1 = tuple(map(operator.add, point_1, offset))
    point_2 = tuple(map(operator.add, point_2, offset))
    point_3 = tuple(map(operator.add, point_3, offset))
    point_0 = (int(point_0[0])-15, int(point_0[1]+10))
    point_1 = (int(point_1[0])-15, int(point_1[1]-10))
    point_2 = (int(point_2[0])+10, int(point_2[1]-10))
    point_3 = (int(point_3[0])+10, int(point_3[1]+10))

    cv2.line(imgOriginalScene, point_0, point_1, SCALAR_GREEN, 2)  # draw 4 red lines
    cv2.line(imgOriginalScene, point_1, point_2, SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, point_2, point_3, SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, point_3, point_0, SCALAR_GREEN, 2)
