import numpy as np
from config import *
from constant import *
from utils import *


def get_base_axis(finger_normal):
    """
    Description: Get base axis, which is the cross product of unit normal vectors of robot and finger
                 See also in robot_model.jpg, where ZG = (0, 0, 1) represents robot normal vector
                 Vector with length a represents finger normal vector
    Input:
        finger_normal: unit normal vector of finger, (3,)
    Output:
        unit direction vector of base axis Z1, (3,)
    """
    return np.cross(finger_normal, np.array([0, 0, 1]))


def get_pivot_axis(finger_normal, base_angle):
    """
    Description: Get pivot axis by 3D triangle geometry
    Input:
        finger_normal: unit normal vector of finger, (3,)
        base_angle: base angle w.r.t vertical position, scalar
    Output:
        unit direction vector of pivot axis Z2, (3,)
    """
    pivot_axis = np.array([finger_normal[0] * np.cos(base_angle),  # projection to get rotated x-comp
                           finger_normal[1] * np.cos(base_angle),  # projection to get rotated y-comp
                           np.sin(base_angle)])  # projection to get rotated z-comp
    return normalize_vec(pivot_axis)


def get_finger_direction(base_axis, pivot_axis):
    """
    Description: Get unit direction vector from base to roller
                 Which is the cross product of pivot axis Z2 and base axis Z1
    Input:
        base_axis: unit direction vector of base axis Z1, (3,)
        pivot_axis: unit direction vector of pivot axis Z2, (3,)
    Output:
        unit direction vector from base to roller, (3,)
    """
    finger_direction = np.cross(pivot_axis, base_axis)
    if finger_direction[2] < 0:
        finger_direction = -finger_direction  # ensure the direction points upward
    return normalize_vec(finger_direction)


def get_roller_pos(base_pos, finger_dir, finger_length):
    """
    Description: Get roller position from relative position between base and roller
    Input:
        base_pos: base position, (3,)
        finger_dir: unit direction vector from base to roller, (3,)
        finger_length: distance between base and roller, scalar
    """
    return base_pos + finger_length * finger_dir


def get_rotating_axis(cur_quat, target_quat):
    """
    Description: Calculate object rotating axis
    Input:
        cur_quat:   object current quaternion, (4,)
        target_quat:   object target quaternion, (4,)
    Output:
        target_axis: object rotating axis, (3,)
    """
    cur_rot_matrix = R.from_quat(mjcquat_to_sciquat(cur_quat)).as_matrix()
    target_rot_matrix = R.from_quat(mjcquat_to_sciquat(target_quat)).as_matrix()
    relative_rot_matrix = np.matmul(np.transpose(cur_rot_matrix), target_rot_matrix)
    axis_angle = R.from_matrix(relative_rot_matrix).as_rotvec()
    target_axis_local = normalize_vec(axis_angle)
    target_axis = np.matmul(cur_rot_matrix, target_axis_local)
    return target_axis


def get_contact_rotation_velocity(roller_pos, roller_r, obj_pos, obj_axis):
    """
    Description: Calculate velocity caused by object rotation at the contact point
    Input:
        roller_pos: position of roller,        (3,)
        roller_r:   radius of roller,          scalar
        obj_pos:    position of object,        (3,)
        obj_axis:   rotation axis of object,   (3,)
    Output:
        vc_w: velocity at contact point due to object rotation, (3,)
    """
    direction_ro = normalize_vec(obj_pos - roller_pos)  # unit direction vector of roller center to object center
    contact_pos = roller_pos + roller_r * direction_ro  # position of contact point
    vector_co = obj_pos - contact_pos  # vector of contact point to object center
    vc_w = np.cross(vector_co, obj_axis)  # velocity at contact point due to object rotation

    return vc_w, contact_pos


def get_base_and_roller_velocity(pivot_axis, roller_pos, obj_pos, vc):
    """
    Description: Get desired base and roller velocity for manipulating the object
                 See also in contact_velocity.jpg
    Input:
        pivot_axis: unit direction vector of pivot axis Z2, (3,)
        roller_pos: roller position, (3,)
        obj_pos: object position, (3,)
        vc: velocity of contact point, (3,)
    Output:
        vc_b: desired base velocity, (3,)
        vc_r: desired roller velocity, (3,)
    """
    if np.linalg.norm(vc) == 0:
        vc_b, vc_r = np.array([0, 0, 0]), np.array([0, 0, 0])
    else:
        # Get vc_b and vc_r direction
        contact_normal = normalize_vec(obj_pos - roller_pos)
        vc_direction = normalize_vec(vc)
        vc_b_direction = pivot_axis
        vc_r_direction = normalize_vec(np.cross(np.cross(vc_b_direction, vc_direction), contact_normal))

        # Solve triangle to get vc_b and vc_r
        # Note that vc_r may equal to zero
        if np.linalg.norm(vc_r_direction) != 0:
            vc_b, vc_r = get_non_orthogonal_projections(vc, vc_b_direction, vc_r_direction)
        else:
            vc_b, vc_r = vc_direction, np.array([0, 0, 0])
    return vc_b, vc_r


def get_non_orthogonal_projections(v1, v2_direction, v3_direction):
    """
    Description: Solve vc_b and vc_r when vc, vc_b direction and vc_r direction are known
                 And vc_b and vc_r are not orthogonal
                 vc = alpha * vc_b + beta * vc_r
    Input:
        v1: velocity 1, (3,)
        v2_direction: unit direction vector of velocity 2, (3,)
        v3_direction: unit direction vector of velocity 3, (3,)
    Output:
        v2: velocity 2, (3,)
        v3: velocity 3, (3,)
    """
    n_z = normalize_vec(np.cross(v2_direction, v3_direction))  # normal to all three vectors
    alpha = np.dot(n_z, np.cross(v3_direction, v1)) / np.dot(n_z, np.cross(v3_direction, v2_direction))
    beta = np.dot(n_z, np.cross(v2_direction, v1)) / np.dot(n_z, np.cross(v2_direction, v3_direction))
    v2 = alpha * v2_direction
    v3 = beta * v3_direction
    return v2, v3


def get_roller_axis(base_axis, pivot_axis, vc_r, cur_pivot_angle, is_palm=False):
    """
    Description: Get desired roller_axis and pivot angle for manipulating the object
    Input:
        base_axis: unit direction vector of base axis Z1, (3,)
        pivot_axis: unit direction vector of pivot axis Z2, (3,)
        vc_r: roller velocity at contact point, (3,)
    Output:
        roller_axis: desired roller axis, (3,)
        pivot_angle: desired pivot angle, scalar
    """
    roller_axis = normalize_vec(np.cross(pivot_axis, vc_r))
    index = 0 if is_palm else 2

    pivot_zero_direction = normalize_vec(np.cross(base_axis, pivot_axis))
    pivot_angle_mag = get_angle(roller_axis, pivot_zero_direction)
    if not is_palm:
        pivot_angle_sign = np.sign(np.dot(np.cross(pivot_zero_direction, roller_axis), -pivot_axis))
    else:
        pivot_angle_sign = np.sign(np.dot(np.cross(pivot_zero_direction, roller_axis), pivot_axis))
    pivot_angle = pivot_angle_mag * pivot_angle_sign

    angle_to_move = abs(cur_pivot_angle - pivot_angle)

    if angle_to_move > np.pi / 2 or (angle_to_move == np.pi / 2 and roller_axis[index] < 0):
        roller_axis *= -1
        pivot_angle_mag = get_angle(roller_axis, pivot_zero_direction)
        if not is_palm:
            pivot_angle_sign = np.sign(np.dot(np.cross(pivot_zero_direction, roller_axis), -pivot_axis))
        else:
            pivot_angle_sign = np.sign(np.dot(np.cross(pivot_zero_direction, roller_axis), pivot_axis))
        pivot_angle = pivot_angle_mag * pivot_angle_sign

    return roller_axis, pivot_angle


def calc_finger_base_init_qpos(obj_pos, obj_r, roller_r, base_pos, finger_normal, finger_length):
    base_angle = MAX_ABS_BASE
    for base_angle in np.linspace(MAX_ABS_BASE, MIN_ABS_BASE):
        base_axis = get_base_axis(finger_normal)
        pivot_axis = get_pivot_axis(finger_normal, base_angle)
        finger_dir = get_finger_direction(base_axis, pivot_axis)
        roller_pos = get_roller_pos(base_pos, finger_dir, finger_length)
        if np.linalg.norm(obj_pos - roller_pos) < obj_r + roller_r:
            break

    return base_angle


def calc_palm_base_init_qpos(obj_pos, obj_r, roller_r, base_pos, palm_length):
    base_height = MIN_ABS_PALM
    for base_height in np.linspace(MIN_ABS_PALM, MAX_ABS_PALM):
        roller_pos = base_pos + np.array([0.0, 0.0, palm_length + base_height])
        if np.linalg.norm(obj_pos - roller_pos) < obj_r + roller_r:
            break

    return base_height


def calc_robot_base_init_qpos(obj_pos, obj_r):
    base_qpos = []
    for finger_num in [0, 1, 2]:
        base_pos = FINGER_BASE_POS[finger_num] + ROBOT_POS
        finger_normal = FINGER_NORMAL[finger_num]
        angle = calc_finger_base_init_qpos(obj_pos, obj_r, ROLLER_RADIUS, base_pos, finger_normal, FINGER_LENGTH)
        base_qpos.append(angle)

    base_pos = PALM_BASE_POS + ROBOT_POS
    height = calc_palm_base_init_qpos(obj_pos, obj_r, ROLLER_RADIUS, base_pos, PALM_LENGTH)
    base_qpos.append(height)

    return np.array(base_qpos)


def calc_robot_init_qpos(obj_frame, obj_r, preset=np.pi / 180.0 * np.zeros(12), transl=np.array([0., 0., 0.]),
                         grasp_near=False, obj_type=OBJ_TYPE, precision=0.001, target_point=np.array([0., 0., 0.2])):
    robot_init_qpos = preset.copy()
    rot_mat = R.from_quat(mjcquat_to_sciquat(obj_frame[-4:])).as_matrix()
    T02 = np.eye(4)
    T02[:3, :3] = rot_mat
    T02[:3, 3] = obj_frame[:3]
    if not grasp_near:
        T21 = np.eye(4)
        T21[:3, 3] = -transl
        T01 = T02 @ T21
        robot_init_qpos[0::3] = calc_robot_base_init_qpos(obj_pos=T01[:3, 3], obj_r=obj_r)
    else:
        if obj_type == "cube":
            x_range, y_range, z_range = [[0., 0.]], [[0., 0.]], [[0., 0.]]
        elif obj_type == "rectangular2":
            x_range, y_range, z_range = [[0., 0.]], [[0., 0.]], [[-0.034, 0.034]]
        elif obj_type == "rectangular":
            x_range, y_range, z_range = [[-0.01, 0.01]], [[-0.02, 0.02]], [[0., 0.]]
        else:
            raise ValueError("Unknown object type!")
        candidate_points, distance = [], []
        for x, y, z in zip(x_range, y_range, z_range):
            points = np.meshgrid(np.linspace(x[0], x[1], round((x[1] - x[0]) / precision + 1)),
                                     np.linspace(y[0], y[1], round((y[1] - y[0]) / precision + 1)),
                                     np.linspace(z[0], z[1], round((z[1] - z[0]) / precision + 1)),
                                     [1])
            points = T02 @ np.array(points).reshape(4, -1)
            points = points.T[:, :3]
            d = np.linalg.norm(points - target_point, axis=1)
            d_min = np.min(d)
            near_point = points[np.argmin(d)]
            candidate_points.append(near_point)
            distance.append(d_min)
        nearest_pos = candidate_points[np.argmin(np.array(distance))]
        robot_init_qpos[0::3] = calc_robot_base_init_qpos(obj_pos=nearest_pos, obj_r=obj_r)

    return robot_init_qpos


def get_virtual_obj_pos(hand, qpos):
    roller_pos = []
    for i in range(3):
        finger = hand[i]
        base_angle = qpos[3 * i]
        pivot_axis = get_pivot_axis(finger.finger_normal, base_angle)
        finger_direction = get_finger_direction(finger.base_axis, pivot_axis)
        roller_pos.append(get_roller_pos(finger.base_pos, finger_direction, finger.finger_length))
    palm_base_height = qpos[9]
    roller_pos.append(hand[3].base_pos + np.array([0.0, 0.0, hand[3].palm_length + palm_base_height]))
    return get_sphere_center(np.array(roller_pos))

