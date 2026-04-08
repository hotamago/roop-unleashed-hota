import numpy as np


class FaceProxy(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def serialize_face(face):
    data = {}
    for key in ("bbox", "kps", "landmark_2d_106", "landmark_2d_68"):
        value = getattr(face, key, None)
        if value is None:
            try:
                value = face[key]
            except Exception:
                value = None
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            data[key] = value.astype(np.float32, copy=False)
        else:
            data[key] = np.asarray(value, dtype=np.float32)
    matrix = getattr(face, "matrix", None)
    if matrix is not None:
        data["matrix"] = matrix.astype(np.float32, copy=False) if isinstance(matrix, np.ndarray) else np.asarray(matrix, dtype=np.float32)
    if hasattr(face, "sex"):
        data["sex"] = face.sex
    landmark_score = getattr(face, "landmark_2d_68_score", None)
    if landmark_score is not None:
        data["landmark_2d_68_score"] = float(landmark_score)
    return data


def deserialize_face(data):
    face = FaceProxy()
    for key in ("bbox", "kps", "landmark_2d_106", "landmark_2d_68", "embedding", "matrix"):
        value = data.get(key)
        if value is None:
            continue
        face[key] = np.array(value, dtype=np.float32)
    if "sex" in data:
        face["sex"] = data["sex"]
    if "landmark_2d_68_score" in data:
        face["landmark_2d_68_score"] = float(data["landmark_2d_68_score"])
    return face


__all__ = ["FaceProxy", "deserialize_face", "serialize_face"]
