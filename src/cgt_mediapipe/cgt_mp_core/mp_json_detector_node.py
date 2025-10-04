from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Optional, Tuple

from ...cgt_core.cgt_patterns import cgt_nodes


class DetectorJsonNode(cgt_nodes.InputNode):
    def __init__(self, json_body_path: str):
        # Json Body is list of dicts
        # dict contains keys: "pose_world_landmarks", "right_hand_world_landmarks", "left_hand_world_landmarks", "face_world_landmarks"
        # and values: list of landmarks (list of [x, y, z])
        self.json_body: list[dict[str, list[list[float]]]] = json.load(
            open(json_body_path)
        )
        self.length = len(self.json_body)
        self.current_frame = 0

    def update(self, data: None, frame: int) -> Tuple[Optional[Any], int]:
        return self.exec_detection(), frame

    @abstractmethod
    def detected_data(self, mp_res: dict[str, list[list[float]]]) -> Any:
        pass

    def exec_detection(self):
        if self.current_frame < self.length:
            mp_res = self.json_body[self.current_frame]
            self.current_frame += 1
            return self.detected_data(mp_res)
        else:
            return None

    def cvt2landmark_array(self, landmark_list: list[list[float]]):
        """landmark_list: A normalized landmark list proto message to be annotated on the image."""
        # list of [landmark_idx, [x, y, z]]
        return [
            [idx, [landmark[0], landmark[1], landmark[2]]]
            for idx, landmark in enumerate(landmark_list)
        ]
