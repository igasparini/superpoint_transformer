import numpy as np


########################################################################
#                              Data splits                             #
########################################################################

# The validation set was arbitrarily chosen:
TILES = {
    'train': [
        'nivo_lidar_clipped_train',
    ],

    'val': [],

    'test': [
        'nivo_lidar_clipped_test',
    ]}

########################################################################
#                                Labels                                #
########################################################################

DALES_NUM_CLASSES = 8

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])

CLASS_NAMES = [
    'Ground',
    'Vegetation',
    'Cars',
    'Trucks',
    'Power lines',
    'Fences',
    'Poles',
    'Buildings',
    'Unknown']

CLASS_COLORS = np.asarray([
    [243, 214, 171],  # sunset
    [ 70, 115,  66],  # fern green
    [233,  50, 239],
    [243, 238,   0],
    [190, 153, 153],
    [  0, 233,  11],
    [239, 114,   0],
    [214,   66,  54],  # vermillon
    [  0,   8, 116]])

# For instance segmentation
MIN_OBJECT_SIZE = 100
THING_CLASSES = [2, 3, 4, 5, 6, 7]
STUFF_CLASSES = [i for i in range(DALES_NUM_CLASSES) if not i in THING_CLASSES]
