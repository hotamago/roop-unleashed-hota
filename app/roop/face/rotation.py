import numpy as np


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


def rotate_image_180(image):
    return np.flip(image, 0)


__all__ = [
    "rotate_anticlockwise",
    "rotate_clockwise",
    "rotate_image_180",
    "rotate_image_90",
]
