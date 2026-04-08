import cv2
import numpy as np

from roop.face.analyser import get_face_analyser
from roop.face.geometry import clamp_cut_values, resize_image_keep_content
from roop.media.capturer import get_video_frame


def get_first_face(frame):
    try:
        faces = get_face_analyser().get(frame)
        return min(faces, key=lambda x: x.bbox[0])
    except Exception:
        return None


def get_all_faces(frame):
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except Exception:
        return None


def extract_face_images(source_filename, video_info, extra_padding=-1.0):
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    for face in faces:
        (start_x, start_y, end_x, end_y) = face["bbox"].astype("int")
        start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, source_image)
        if extra_padding > 0.0:
            if source_image.shape[:2] == (512, 512):
                face_data.append([face, source_image])
                continue

            found = False
            for attempt in range(1, 3):
                (start_x, start_y, end_x, end_y) = face["bbox"].astype("int")
                start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, source_image)
                cutout_padding = extra_padding
                padding = int((end_y - start_y) * cutout_padding)
                old_y = start_y
                start_y -= padding

                factor = 0.25 if attempt == 1 else 0.5
                cutout_padding = factor
                padding = int((end_y - old_y) * cutout_padding)
                end_y += padding
                padding = int((end_x - start_x) * cutout_padding)
                start_x -= padding
                end_x += padding
                start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, source_image)
                face_temp = source_image[start_y:end_y, start_x:end_x]
                face_temp = resize_image_keep_content(face_temp)
                testfaces = get_all_faces(face_temp)
                if testfaces is not None and len(testfaces) > 0:
                    face_data.append([testfaces[0], face_temp])
                    found = True
                    break

            if not found:
                print("No face found after resizing, this shouldn't happen!")
            continue

        face_temp = source_image[start_y:end_y, start_x:end_x]
        if face_temp.size < 1:
            continue

        face_data.append([face, face_temp])
    return face_data


__all__ = ["extract_face_images", "get_all_faces", "get_first_face"]
