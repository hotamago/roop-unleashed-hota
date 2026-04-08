from typing import Any

import numpy
from insightface.app.common import Face

from roop.pipeline.faceset import FaceSet

Face = Face
FaceSet = FaceSet
Frame = numpy.ndarray[Any, Any]

__all__ = ["Face", "FaceSet", "Frame"]
