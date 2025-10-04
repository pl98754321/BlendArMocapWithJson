import time

from .mp_json_detector_node import DetectorJsonNode


class HolisticDetectorJson(DetectorJsonNode):
    def __init__(
        self,
        json_body_path: str,
    ):
        DetectorJsonNode.__init__(self, json_body_path)

    def empty_data(self):
        return [[[], []], [[[]]], []]

    def detected_data(self, mp_res):
        face, pose, l_hand, r_hand = [], [], [], []
        pose = self.cvt2landmark_array(mp_res["pose_world_landmarks"])
        # face = self.cvt2landmark_array(mp_res["face_world_landmarks"])
        l_hand = self.cvt2landmark_array(mp_res["left_hand_world_landmarks"])
        r_hand = self.cvt2landmark_array(mp_res["right_hand_world_landmarks"])
        # TODO: recheck every update, mp hands are flipped while detecting holistic.
        return [[[l_hand], [r_hand]], face, pose]

    def contains_features(self, mp_res: dict[str, list[list[float]]]):
        return "pose_world_landmarks" in mp_res.keys()
