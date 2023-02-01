import numpy as np

from config import *
from utils import *
from heuristic import *
from constant import *


class Finger(object):
    def __init__(self, finger_name):
        finger_dict = {'front': 0, 'left': 1, 'right': 2}
        self.finger_num = finger_dict[finger_name]
        self.finger_length = FINGER_LENGTH
        self.finger_normal = FINGER_NORMAL[self.finger_num]
        self.base_pos = FINGER_BASE_POS[self.finger_num] + ROBOT_POS
        self.base_axis = get_base_axis(self.finger_normal)
        self.roller_r = ROLLER_RADIUS

    def calibrate(self, observation, target_pos, target_quat, virtual_obj_pos=None, vc_v_scale=0.006, vc_w_scale=0.010):
        cur_pos = observation[-7:-4]
        cur_quat = observation[-4:]
        calibration_pos = target_pos - cur_pos
        if np.linalg.norm(calibration_pos) == 0.0:
            self.vc_v = np.array([0.0, 0.0, 0.0])
        else:
            self.vc_v = normalize_vec(calibration_pos)
        self.base_angle = observation[3 * self.finger_num]
        self.pivot_axis = get_pivot_axis(self.finger_normal, self.base_angle)
        self.finger_direction = get_finger_direction(self.base_axis, self.pivot_axis)
        self.roller_pos = get_roller_pos(self.base_pos, self.finger_direction, self.finger_length)
        obj_pos = observation[-7:-4] if virtual_obj_pos is None else virtual_obj_pos

        target_axis = get_rotating_axis(cur_quat, target_quat)
        self.vc_w, self.contact_pos = get_contact_rotation_velocity(self.roller_pos, self.roller_r,
                                                                    obj_pos, target_axis)
        pos_err = get_pos_err(cur_pos, target_pos)
        quat_err = get_quat_err(cur_quat, target_quat)

        self.vc = vc_v_scale * self.vc_v * pos_err + vc_w_scale * self.vc_w * quat_err

        if np.linalg.norm(self.vc) == 0.0:
            self.pivot_angle = observation[3 * self.finger_num + 1]
            self.delta_base, self.delta_roller = 0.0, 0.0
            return

        if np.linalg.norm(np.cross(self.vc, self.pivot_axis)) == 0.0:
            self.vc_b, self.vc_r = self.vc, np.array([0.0, 0.0, 0.0])
            self.pivot_angle = observation[3 * self.finger_num + 1]
            self.delta_roller = 0.0
        else:
            self.vc_b, self.vc_r = get_base_and_roller_velocity(self.pivot_axis, self.roller_pos, obj_pos, self.vc)
            self.roller_axis, self.pivot_angle = get_roller_axis(self.base_axis, self.pivot_axis, self.vc_r,
                                                                 observation[3 * self.finger_num + 1])
            self.r_effective_roller = distance_between_lines(self.roller_pos, self.roller_axis, self.contact_pos,
                                                             self.vc_r)
            self.delta_roller = np.linalg.norm(self.vc_r) / self.r_effective_roller

        if np.linalg.norm(self.vc_b) == 0.0:
            self.delta_base = 0.0
        else:
            self.r_effective_base = distance_between_lines(self.base_pos, self.base_axis, self.contact_pos, self.vc_b)
            self.delta_base = np.linalg.norm(self.vc_b) / self.r_effective_base


class Palm(object):
    def __init__(self):
        self.palm_length = PALM_LENGTH
        self.base_pos = PALM_BASE_POS + ROBOT_POS
        self.base_axis = np.array([0.0, 1.0, 0.0])
        self.roller_r = ROLLER_RADIUS

    def calibrate(self, observation, target_pos, target_quat, virtual_obj_pos=None, vc_v_scale=0.006, vc_w_scale=0.010):
        cur_pos = observation[-7:-4]
        cur_quat = observation[-4:]
        calibration_pos = target_pos - cur_pos
        if np.linalg.norm(calibration_pos) == 0.0:
            self.vc_v = np.array([0.0, 0.0, 0.0])
        else:
            self.vc_v = normalize_vec(calibration_pos)
        self.base_height = observation[9]
        self.pivot_axis = np.array([0.0, 0.0, 1.0])
        self.roller_pos = self.base_pos + np.array([0.0, 0.0, self.palm_length + self.base_height])
        obj_pos = observation[-7:-4] if virtual_obj_pos is None else virtual_obj_pos
        target_axis = get_rotating_axis(cur_quat, target_quat)
        self.vc_w, self.contact_pos = get_contact_rotation_velocity(self.roller_pos, self.roller_r,
                                                                    obj_pos, target_axis)
        pos_err = get_pos_err(cur_pos, target_pos)
        quat_err = get_quat_err(cur_quat, target_quat)

        self.vc = vc_v_scale * self.vc_v * pos_err + vc_w_scale * self.vc_w * quat_err

        if np.linalg.norm(self.vc) == 0.0:
            self.pivot_angle = observation[10]
            self.delta_base, self.delta_roller = 0.0, 0.0
            return

        if np.linalg.norm(np.cross(self.vc, self.pivot_axis)) == 0.0:
            self.vc_b, self.vc_r = self.vc, np.array([0.0, 0.0, 0.0])
            self.pivot_angle = observation[10]
            self.delta_roller = 0.0
        else:
            self.vc_b, self.vc_r = get_base_and_roller_velocity(self.pivot_axis, self.roller_pos, obj_pos, self.vc)
            self.roller_axis, self.pivot_angle = get_roller_axis(self.base_axis, self.pivot_axis, self.vc_r, observation[10],
                                                                 is_palm=True)

            self.r_effective_roller = distance_between_lines(self.roller_pos, self.roller_axis, self.contact_pos,
                                                             self.vc_r)
            self.delta_roller = np.linalg.norm(self.vc_r) / self.r_effective_roller

        self.delta_base = self.vc_b[2]
