import torch
import numpy as np

NYU_class_names = [
    "empty",
    "ceiling",
    "floor",
    "wall",
    "window",
    "chair",
    "bed",
    "sofa",
    "table",
    "tvs",
    "furn",
    "objs",
]
class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

class_freq_1_4 = np.array(
    [
        43744234,
        80205,
        1070052,
        905632,
        116952,
        180994,
        436852,
        279714,
        254611,
        28247,
        1805949,
        850724,
    ]
)
class_freq_1_8 = np.array(
    [
        5176253,
        17277,
        220105,
        183849,
        21827,
        33520,
        67022,
        44248,
        46615,
        4419,
        290218,
        142573,
    ]
)
class_freq_1_16 = np.array(
    [587620, 3820, 46836, 36256, 4241, 5978, 10939, 8000, 8224, 781, 49778, 25864]
)

NYU_CAM_K = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]], dtype=np.float32)
NYU_VOXEL_SIZE = 0.08
NYU_IMG_W = 640
NYU_IMG_H = 480
NYU_SCENE_SIZE = (4.8, 4.8, 2.88)
