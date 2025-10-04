import time

from .mp_json_detector_node import DetectorJsonNode


class PoseDetectorJson(DetectorJsonNode):
    def __init__(
        self,
        json_body_path: str,
    ):
        DetectorJsonNode.__init__(self, json_body_path)

    def detected_data(self, mp_res: dict[str, list[list[float]]]):
        return self.cvt2landmark_array(mp_res["pose_world_landmarks"])

    def contains_features(self, mp_res: dict[str, list[list[float]]]):
        return "pose_world_landmarks" in mp_res.keys()
