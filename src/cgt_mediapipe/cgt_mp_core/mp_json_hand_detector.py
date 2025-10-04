from .mp_json_detector_node import DetectorJsonNode


class HandDetectorJson(DetectorJsonNode):
    def __init__(self, json_body_path: str):
        DetectorJsonNode.__init__(self, json_body_path)

    def detected_data(self, mp_res: dict[str, list[list[float]]]):
        left_hand_data = self.cvt2landmark_array(mp_res["left_hand_world_landmarks"])
        right_hand_data = self.cvt2landmark_array(mp_res["right_hand_world_landmarks"])
        return [[left_hand_data], [right_hand_data]]

    def contains_features(self, mp_res: dict[str, list[list[float]]]):
        if ("left_hand_world_landmarks" in mp_res.keys()) and (
            "right_hand_world_landmarks" in mp_res.keys()
        ):
            return True
        return False
