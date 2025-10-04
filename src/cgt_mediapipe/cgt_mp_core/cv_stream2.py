from __future__ import annotations

import json
import logging
import time
from typing import Tuple, Union

import cv2
import numpy as np


class StreamJson:
    updated: bool = None
    mp_res: np.ndarray = None

    def __init__(self, json_body_path: Union[str, int]):
        self.json_body = json.load(open(json_body_path))
        self.length = len(self.json_body)
        self.current_frame = 0

    def update(self):
        if self.current_frame < self.length:
            self.updated = True
            self.mp_res = self.json_body[self.current_frame]
            self.current_frame += 1
        else:
            self.updated = False

    def exit_stream(self):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.debug("ATTEMPT TO EXIT STEAM")
            return True
        else:
            return False
