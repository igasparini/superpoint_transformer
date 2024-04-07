import numpy as np


########################################################################
#                         Download information                         #
########################################################################

FORM_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSefhHMMvN0Uwjnj_vWQgYSvtFOtaoGFWsTIcRuBTnP09NHR7A/viewform?fbzx=5530674395784263977'

# DALES in LAS format
LAS_TAR_NAME = 'dales_semantic_segmentation_las.tar.gz'
LAS_UNTAR_NAME = "dales_las"

# DALES in PLY format
PLY_TAR_NAME = 'dales_semantic_segmentation_ply.tar.gz'
PLY_UNTAR_NAME = "dales_ply"

# DALES in PLY, only version with intensity and instance labels
OBJECTS_TAR_NAME = 'DALESObjects.tar.gz'
OBJECTS_UNTAR_NAME = "DALESObjects"


########################################################################
#                              Data splits                             #
########################################################################

# The validation set was arbitrarily chosen:
TILES = {
    'train': [
        'birmingham_block_0',
        'birmingham_block_1',
        'birmingham_block_4',
        'birmingham_block_5',
        'birmingham_block_6',
        'birmingham_block_9',
        'birmingham_block_10',
        'birmingham_block_11',
        'birmingham_block_12',
        'birmingham_block_13',
        #'cambridge_block_0', Buggy. See https://github.com/QingyongHu/SensatUrban/issues/4
        #'cambridge_block_1', Buggy. See https://github.com/QingyongHu/SensatUrban/issues/4
        'cambridge_block_2',
        'cambridge_block_3',
        'cambridge_block_6',
        'cambridge_block_7',
        'cambridge_block_8',
        'cambridge_block_9',
        'cambridge_block_12',
        #'cambridge_block_13', Ground issue with RANSAC
        'cambridge_block_14',
        'cambridge_block_17',
        'cambridge_block_18',
        'cambridge_block_19',
        'cambridge_block_20',
        'cambridge_block_21',
        'cambridge_block_23',
        'cambridge_block_25',
        'cambridge_block_26',
        'cambridge_block_33',
        #'cambridge_block_34' Buggy. See https://github.com/QingyongHu/SensatUrban/issues/4
    ],

    'val': [
        'birmingham_block_3',
        'birmingham_block_7',
        'cambridge_block_4',
        'cambridge_block_10',
        'cambridge_block_28'
    ],

    'test': [
        'cambridge_block_32'
    ]}

########################################################################
#                                Labels                                #
########################################################################

SENSAT_NUM_CLASSES = 13

ID2TRAINID = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

CLASS_NAMES = [
    'Ground',
    'Vegetation',
    'Buildings',
    'Wall',
    'Bridge',
    'Parking',
    'Rail',
    'Traffic Road',
    'Street Furniture',
    'Car',
    'Footpath',
    'Bike',
    'Water']

CLASS_COLORS = np.asarray([
    [ 84,  50,  25], # Ground
    [ 70, 115,  66], # vegetation - fern green
    [122,  18,  18], # Building
    [114, 114, 114], # Wall
    [219,  83, 217], # Bridge
    [  1,   6, 140], # Parking
    [  0,   0,   0], # Rail
    [ 81, 250,   2], # Traffic Road
    [250, 163,   2], # Street Furniture
    [ 67, 122,  82], # Car
    [ 99,  80,  36], # Footpath
    [230, 250,   7], # Bike
    [ 35, 194, 247], # Water
    ])

# # For instance segmentation
# MIN_OBJECT_SIZE = 100
THING_CLASSES = [0, 1, 12]
STUFF_CLASSES = [i for i in range(SENSAT_NUM_CLASSES) if not i in THING_CLASSES]
