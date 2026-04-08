import cv2
import numpy as np


def clamp_cut_values(start_x, end_x, start_y, end_y, image):
    if start_x < 0:
        start_x = 0
    if end_x > image.shape[1]:
        end_x = image.shape[1]
    if start_y < 0:
        start_y = 0
    if end_y > image.shape[0]:
        end_y = image.shape[0]
    return start_x, end_x, start_y, end_y


def face_offset_top(face, offset):
    face["bbox"][1] += offset
    face["bbox"][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face["landmark_2d_106"] = lm106 + add
    return face


def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (height, width) = image.shape[:2]
    if height > width:
        ratio = new_height / float(height)
        dim = (int(width * ratio), new_height)
    else:
        ratio = new_width / float(width)
        dim = (new_width, int(height * ratio))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (height, width) = image.shape[:2]
    if height == new_height and width == new_width:
        return image
    resized = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - width) if height == new_height else (new_height - height)
    start_offs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if height == new_height:
        resized[0:new_height, start_offs : new_width - offs] = image
    else:
        resized[start_offs : new_height - offs, 0:new_width] = image
    return resized


def create_blank_image(width, height):
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:] = [0, 0, 0, 0]
    return image


def cutout(frame, start_x, start_y, end_x, end_y):
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if end_x > frame.shape[1]:
        end_x = frame.shape[1]
    if end_y > frame.shape[0]:
        end_y = frame.shape[0]
    return frame[start_y:end_y, start_x:end_x], start_x, start_y, end_x, end_y


def paste_simple(src, dest, start_x, start_y):
    end_x = start_x + src.shape[1]
    end_y = start_y + src.shape[0]
    start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, dest)
    dest[start_y:end_y, start_x:end_x] = src
    return dest


def simple_blend_with_mask(image1, image2, mask):
    blended_image = image1.astype(np.float32) * (1.0 - mask) + image2.astype(np.float32) * mask
    return blended_image.astype(np.uint8)


__all__ = [
    "clamp_cut_values",
    "create_blank_image",
    "cutout",
    "face_offset_top",
    "paste_simple",
    "resize_image_keep_content",
    "simple_blend_with_mask",
]
