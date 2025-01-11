from enum import Enum

class LeakageTypes(object):
    AIR_LEAK_TOP = "Air leakage at top"
    AIR_LEAK_BOTTOM = "Air leakage at bottom"
    AIR_LEAK_LEFT = "Air leakage at left"
    AIR_LEAK_RIGHT = "Air leakage at right"
    POOR_INSULATION_SMALL = "Small area of poor insulation"
    POOR_INSULATION_LARGE = "Large area of poor insulation"
    WINDOW_AJAR = "Air leakage due to ajar window"

class Camera(object):
    E50 = "E50"
    EDGEPRO = "EdgePro"
    THERMAL_IMAGE_TYPE = "tiff"
    E50_THERMAL_HORIZONTAL_FOV = 25
    E50_THERMAL_VERTICAL_FOV = 19
    E50_OPTICAL_HORIZONTAL_FOV = 53
    E50_OPTICAL_VERTICAL_FOV = 41

class Annotation(Enum):
    IMAGE_NAME = 1
    DATE_TAKEN = 2
    LEAK_SIGNATURE = 3
    WINDOW_NAME = 4
    INSIDE_TEMP = 5
    OUTSIDE_TEMP = 6
    MIN_TEMP = 7
    MAX_TEMP = 8
