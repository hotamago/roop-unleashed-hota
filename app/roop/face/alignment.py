import cv2
import numpy as np
from skimage import transform as trans


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio
    else:
        ratio = float(image_size) / 112.0
        diff_x = 0

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]


def align_crop(img, landmark, image_size=112, mode="arcface"):
    del mode
    matrix = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, matrix, (image_size, image_size), borderValue=0.0)
    return warped, matrix


def square_crop(im, size):
    if im.shape[0] > im.shape[1]:
        height = size
        width = int(float(im.shape[1]) / im.shape[0] * size)
        scale = float(size) / im.shape[0]
    else:
        width = size
        height = int(float(im.shape[0]) / im.shape[1] * size)
        scale = float(size) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((size, size, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    matrix = t.params[0:2]
    cropped = cv2.warpAffine(data, matrix, (output_size, output_size), borderValue=0.0)
    return cropped, matrix


def trans_points2d(pts, matrix):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for index in range(pts.shape[0]):
        pt = pts[index]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(matrix, new_pt)
        new_pts[index] = new_pt[0:2]
    return new_pts


def trans_points3d(pts, matrix):
    scale = np.sqrt(matrix[0][0] * matrix[0][0] + matrix[0][1] * matrix[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for index in range(pts.shape[0]):
        pt = pts[index]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(matrix, new_pt)
        new_pts[index][0:2] = new_pt[0:2]
        new_pts[index][2] = pts[index][2] * scale
    return new_pts


def trans_points(pts, matrix):
    if pts.shape[1] == 2:
        return trans_points2d(pts, matrix)
    return trans_points3d(pts, matrix)


__all__ = [
    "align_crop",
    "estimate_norm",
    "square_crop",
    "trans_points",
    "trans_points2d",
    "trans_points3d",
    "transform",
]
