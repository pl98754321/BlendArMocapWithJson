import logging
from pathlib import Path
from typing import Optional

import bpy  # type: ignore

from ..cgt_core.cgt_patterns import cgt_nodes


class WM_CGT_MP_modal_json_detection_operator(bpy.types.Operator):
    bl_label = "Json Input Operator"
    bl_idname = "wm.cgt_json_feature_detection_operator"
    bl_description = "Detect solution in Stream."

    _timer: Optional[bpy.types.Timer] = None
    node_chain: Optional[cgt_nodes.NodeChain] = None
    frame: int = 1
    key_step: int = 1
    memo: list

    def get_chain(self) -> Optional[cgt_nodes.NodeChain]:
        from ..cgt_core import cgt_core_chains
        from .cgt_mp_core import (
            mp_json_face_detector,
            mp_json_hand_detector,
            mp_json_holistic_detector,
            mp_json_pose_detector,
        )

        mov_body_data_path = bpy.path.abspath(self.user.mov_body_data_path)
        logging.info(f"Path to mov: {mov_body_data_path}")
        if not Path(mov_body_data_path).is_file():
            self.user.modal_active = False
            logging.error(f"GIVEN PATH IS NOT VALID {mov_body_data_path}")
            return {"FINISHED"}

        node_chain = cgt_nodes.NodeChain()
        move_path = self.user.mov_body_data_path
        if self.user.enum_detection_type == "HAND":
            input_node = mp_json_hand_detector.HandDetectorJson(move_path)
            chain_template = cgt_core_chains.HandNodeChain()
        elif self.user.enum_detection_type == "POSE":
            input_node = mp_json_pose_detector.PoseDetectorJson(move_path)
            chain_template = cgt_core_chains.PoseNodeChain()
        elif self.user.enum_detection_type == "FACE":
            input_node = mp_json_face_detector.FaceDetectorJson(move_path)
            chain_template = cgt_core_chains.FaceNodeChain()
        elif self.user.enum_detection_type == "HOLISTIC":
            input_node = mp_json_holistic_detector.HolisticDetectorJson(move_path)
            chain_template = cgt_core_chains.HolisticNodeChainGroup()
        else:
            logging.error(f"UNKNOWN DETECTION TYPE {self.user.enum_detection_type}")
            self.user.modal_active = False
            return None
        node_chain.append(input_node)
        node_chain.append(chain_template)
        logging.info(f"{node_chain}")
        return node_chain

    def execute(self, context):
        """Runs movie or stream detection depending on user input."""
        self.user = getattr(context.scene, "cgtinker_mediapipe")
        assert self.user is not None

        # don't activate if modal is running
        if self.user.modal_active is True:
            self.user.modal_active = False
            self.report({"INFO"}, "Stopped detection.")
            return {"FINISHED"}
        else:
            self.user.modal_active = True

        # init stream and chain
        self.node_chain = self.get_chain()
        if self.node_chain is None:
            self.user.modal_active = False
            return {"FINISHED"}

        # add a timer property and start running
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.0, window=context.window)
        context.window_manager.modal_handler_add(self)

        # memo skipped frames
        self.memo = []
        self.report({"INFO"}, f"Running {self.user.enum_detection_type} as modal.")
        return {"RUNNING_MODAL"}

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT", "POSE"}

    @staticmethod
    def simple_smoothing(memo, cur):
        """Expects list with sub-lists containing [int, [float, float, float]].
        Smooths the float part of the sub-lists."""

        def smooth_by_add_divide(x, y):
            for i, *_ in enumerate(zip(x, y)):
                x[i] += y[i]
                x[i] /= 2

        def addable(x, y):
            # check if [int, [float, float float]]
            if not isinstance(x, list) or not isinstance(y, list):
                print("CATCHED NOT LIST ERR")
                return False

            if not len(x) == 2 or not len(y) == 2:
                return False

            if not len(x[1]) == 3 or not len(y[1]) == 3:
                return False

            smooth_by_add_divide(x[1], y[1])
            return True

        def smooth_memo_contents(x, y):
            # checks if addable contents, else splits into sub-arrays
            # and retries. y may get added to x, if x is empty.
            if not isinstance(y, list):
                return
            if not isinstance(x, list):
                x = y
            if len(x) == 0 and len(y) != 0:
                x += y

            for l1, l2 in zip(x, y):
                if not addable(l1, l2):
                    smooth_memo_contents(l1, l2)

        smooth_memo_contents(memo, cur)
        return memo

    def modal(self, context, event):
        """Run detection as modal operation, finish with 'Q', 'ESC' or 'RIGHT MOUSE'."""
        assert self.node_chain is not None
        if event.type == "TIMER" and self.user.modal_active:
            if self.user.detection_input_type == "movie":
                # get data
                data, _frame = self.node_chain.nodes[0].update([], self.frame)
                if data is None:
                    return self.cancel(context)

                # smooth gathered data
                self.simple_smoothing(self.memo, data)
                if self.frame % self.key_step == 0:
                    for node in self.node_chain.nodes[1:]:
                        node.update(self.memo, self.frame)
                    self.memo.clear()

                self.frame += 1
            else:
                data, _ = self.node_chain.update([], self.frame)
                if data is None:
                    return self.cancel(context)
                self.frame += self.key_step

        if event.type in {"Q", "ESC", "RIGHT_MOUSE"} or self.user.modal_active is False:
            return self.cancel(context)

        return {"PASS_THROUGH"}

    def cancel(self, context):
        """Upon finishing detection clear the handlers."""
        self.user.modal_active = False  # noqa
        del self.node_chain
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        logging.debug("FINISHED DETECTION")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(WM_CGT_MP_modal_json_detection_operator)


def unregister():
    bpy.utils.unregister_class(WM_CGT_MP_modal_json_detection_operator)
