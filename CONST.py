# Project folder
ROOT_FOLDER = "/Users/sulerhy/Desktop/number-plate-dectect/"
# Input folder
INPUT_FOLDER = "resources/input_images/"
# Output folder
OUTPUT_FOLDER = "resources/output_images/"

MAX_PROPOSED_OBJECTS = 5
# TARGETED_OBJECT: the area that number_plate usually appear in car (here 60% = 0.6 from bottom to top)
TARGETED_OBJECT = 0.6
APRROX_POLY_DP = 0.01

# contour setting
MAX_TAKEN_CONTOUR = 10
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_SIZE = 2
