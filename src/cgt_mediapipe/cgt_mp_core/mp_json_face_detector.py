from typing import Mapping, Tuple

import mediapipe as mp
from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from .mp_json_detector_node import DetectorJsonNode


class FaceDetectorJson(DetectorJsonNode):
    def __init__(
        self,
        json_body_path: str,
    ):
        DetectorJsonNode.__init__(self, json_body_path)

    def empty_data(self):
        return [[[]]]

    def detected_data(self, mp_res: dict[str, list[list[float]]]):
        # list of list of [idx,[x,y,z]]
        return [self.cvt2landmark_array(mp_res["face_world_landmarks"])]

    def contains_features(self, mp_res: dict[str, list[list[float]]]):
        if "face_world_landmarks" not in mp_res.keys():
            return False
        return True
