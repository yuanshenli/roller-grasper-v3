import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '../RGV3'))

from utils import *
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from expert import EnvExpert, get_robot_init_qpos
from heuristic import calc_robot_init_qpos
from constant import *


def get_identical_orientation(obj_type):

    if obj_type in ["cube"]:
        euler_angles = (np.array(np.meshgrid([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3])).T.reshape(-1, 3) * 90.).tolist()
    elif obj_type in ["rectangular2"]:
        euler_angles = (np.array(np.meshgrid([0, 2], [0], [0, 1, 2, 3])).T.reshape(-1, 3) * 90.).tolist()
    elif obj_type in ["rectangular"]:
        euler_angles = (np.array(np.meshgrid([0, 2], [0], [0, 2])).T.reshape(-1, 3) * 90.).tolist()
    elif obj_type in ["cube_w_opening"]:
        euler_angles = (np.array(np.meshgrid([0], [0], [0, 1, 2, 3])).T.reshape(-1, 3) * 90.).tolist()
    elif obj_type in ['mug']:
        euler_angles = np.array([[0., 0., 0.]])
    else:
        euler_angles = np.array([[0., 0., 0.]])

    appeared_quat = []
    unique_eulers = []

    # get rid of euler angles that actually represents the same orientation
    for ea in euler_angles:
        # quat = euler_to_quat(ea, degrees=True).tolist()
        r = R.from_euler("XYZ", ea, degrees=True)
        quat = r.as_quat().tolist()
        # print(quat)
        logged = False
        for quat_old in appeared_quat:
            if get_quat_err(np.array(quat), np.array(quat_old)) < 0.001:
                # print("logged")
                logged = True
                break

        if not logged:
            appeared_quat.append(quat)
            unique_eulers.append(ea)

    [print(euler) for euler in unique_eulers]   # each euler angle here represents an unique orientation
    np.save(obj_type + "_identical_orientation.npy", np.array(unique_eulers))

    return np.array(unique_eulers)

def check_identical_orientation(obj_type, unique_eulers):
    # start_frame = R.random(random_state=360).as_matrix()       # sample a random frame
    # end_frame = R.random(random_state=360).as_matrix()

    start_frame = calc_single_frame(((0.0, 0.0, 0.20), 90, (0, 0, 1)), degrees=True)
    end_frame = calc_single_frame(((0.0, 0.0, 0.20), 60, (1, 2, -1)), degrees=True)
    # e_start = [0., 90., 0.]
    e_start = [0., 45., 45.]
    start_frame[-4:] = euler_to_quat(e_start, degrees=True)
    # e_end = [180., 0., 90.]
    e_end = [45., 45., 0.]
    end_frame[-4:] = euler_to_quat(e_end, degrees=True)

    # axis, theta = get_rel_angle_axis(start_frame[-4:], end_frame[-4:])
    print("start quat: ", mjcquat_to_sciquat(start_frame[-4:]))
    print("end quat: ", mjcquat_to_sciquat(end_frame[-4:]))

    start_frame = R.from_quat(mjcquat_to_sciquat(start_frame[-4:])).as_matrix()
    end_frame = R.from_quat(mjcquat_to_sciquat(end_frame[-4:])).as_matrix()

    relative1 = start_frame.T @ end_frame
    rotvec_global = start_frame @ R.from_matrix(relative1).as_rotvec()

    fig = plt.figure(figsize=(15, 10))
    # check if the resulting frames make sense
    for i, euler in enumerate(unique_eulers):
        print(i, euler)
        new_rotation = R.from_euler("XYZ", euler, degrees=True).as_matrix()
        new_start_matrix = start_frame @ new_rotation
        new_start_quat = sciquat_to_mjcquat(R.from_matrix(new_start_matrix).as_quat())
        rotvec_in_new_start_frame = new_start_matrix.T @ rotvec_global
        new_end_matrix = new_start_matrix @ R.from_rotvec(rotvec_in_new_start_frame).as_matrix()
        new_end_quat = sciquat_to_mjcquat(R.from_matrix(new_end_matrix).as_quat())
        new_start_frame = np.hstack((np.array([0., 0., 0.2]), new_start_quat))
        new_end_frame = np.hstack((np.array([0., 0., 0.2]), new_end_quat))

        robot_init_qpos = get_robot_init_qpos(start_frame=new_start_frame, obj_type=obj_type, render=True)
        # robot_init_qpos = calc_robot_init_qpos(new_start_frame, obj_r=0.03)
        # print("robot_init_qpos", robot_init_qpos)
        env_expert = EnvExpert(robot_init_qpos, new_start_frame, new_end_frame, obj_type=obj_type, render=True)
        env_expert.reset(mode="hard_ctrl")
        while True:
            action = env_expert.get_expert_action(state=None)
            state, reward, done, success, info = env_expert.step(action)
            if done:
                print("Success: {} Error: {:.4f} ".format(success, env_expert.min_err))
                break


def augment_data(obj_type, raw_data_file="collision.npy", augmented_data_file="collision_augmented.npy"):
    unique_eulers = get_identical_orientation(obj_type)
    print(unique_eulers)
    raw_data = np.load(raw_data_file)
    augmented_data = []
    for i, instance in enumerate(raw_data):
        if i % 10000 == 0:
            print(i / raw_data.shape[0] * 100, "%")
        start_pos, start_quat, end_pos, end_quat, label = np.split(instance, [3, 7, 10, 14])
        start_rot_matrix = R.from_quat(mjcquat_to_sciquat(start_quat)).as_matrix()
        end_rot_matrix = R.from_quat(mjcquat_to_sciquat(end_quat)).as_matrix()
        rel_rot_matrix = start_rot_matrix.T @ end_rot_matrix
        rotvec_global = start_rot_matrix @ R.from_matrix(rel_rot_matrix).as_rotvec()
        for euler in unique_eulers:
            new_rotation = R.from_euler("XYZ", euler, degrees=True).as_matrix()
            new_start_matrix = start_rot_matrix @ new_rotation
            new_start_quat = sciquat_to_mjcquat(R.from_matrix(new_start_matrix).as_quat())
            rotvec_in_new_start_frame = new_start_matrix.T @ rotvec_global
            new_end_matrix = new_start_matrix @ R.from_rotvec(rotvec_in_new_start_frame).as_matrix()
            new_end_quat = sciquat_to_mjcquat(R.from_matrix(new_end_matrix).as_quat())
            new_start_axis, new_start_theta = quat_to_rotvec(new_start_quat)
            new_end_axis, new_end_theta = quat_to_rotvec(new_end_quat)
            # new_instance = np.hstack((start_pos, new_start_quat, end_pos, new_end_quat, label))
            new_instance = np.hstack((start_pos, new_start_axis, new_start_theta, end_pos, new_end_axis, new_end_theta, label))
            assert new_instance.shape == (15,)
            augmented_data.append(new_instance)
    augmented_data = np.array(augmented_data)
    print(augmented_data.shape)
    np.save(augmented_data_file, augmented_data)