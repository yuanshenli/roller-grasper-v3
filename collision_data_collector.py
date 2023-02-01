import pandas
import numpy as np
import math
import random
import torch
import time
import os
import argparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from config import *
from heuristic import *
from expert import EnvExpert, get_robot_init_qpos
from rrt_pc import load_path, get_frames_from_path
from sampling import generate_random_case
from augmentation import augment_data


def collect_collision_data(obj_type, obj_r, n_instance,
                           max_pos_dev=np.array([0.02, 0.02, 0.01]), angle_range=np.array([0, 180]),
                           data_file="collision.npy", save_folder="collision_data", render=False,
                           use_contact_points_log=False,
                           sample_near_success=False, success_file=None, pos_noise=0.005, angle_noise=15,
                           success_id_low=None, success_id_high=None,
                           slip_detection=True, dp_thresh=0.006, dq_thresh=11 * np.pi / 180.0, dt_thresh=5,
                           seed_log_file="seed_log.npy", seed_range=(0, int(1e9))):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    xml_path = os.path.join(save_folder, "rgv3.xml")
    collision = []
    robot_init_qpos_log = []
    contact_points_log = []
    if sample_near_success:
        assert success_file is not None
        success_cases = np.load(success_file)
        print("{} success cases loaded.".format(success_cases.shape[0]))
        print("Success id range set to [{}, {})".format(success_id_low, success_id_high))
        if success_id_low >= success_cases.shape[0] or success_id_low >= success_id_high:
            return
        success_cases = success_cases[success_id_low: min(success_cases.shape[0], success_id_high)]
        n_instance = success_cases.shape[0] * 10
    for i in range(n_instance):
        log("[No. {}]".format(i), 'b', 'B')
        if sample_near_success:
            success_case = success_cases[i // 10]
        else:
            success_case = None
        start_frame, end_frame, _ = generate_random_case(obj_type=obj_type, obj_r=obj_r, frame_stability_check=False,
                                                         use_transl=True,
                                                         max_pos_dev=max_pos_dev, angle_range=angle_range,
                                                         sample_near_success=sample_near_success,
                                                         success_case=success_case,
                                                         pos_noise=pos_noise, angle_noise=angle_noise,
                                                         seed_log_file=seed_log_file, seed_range=seed_range)
        log("Start Frame: {}".format(start_frame))
        log("End Frame: {}".format(end_frame))
        # robot_init_qpos = calc_robot_init_qpos(obj_frame=start_frame, obj_type=obj_type, obj_r=0.03, grasp_near=True)
        robot_init_qpos = get_robot_init_qpos(start_frame, obj_type=obj_type, xml_path=xml_path, render=render)
        robot_init_qpos[1::3] = np.random.uniform(MIN_ABS_PIVOT, MAX_ABS_PIVOT, size=(4,))
        robot_init_qpos_log.append(robot_init_qpos.copy())
        env_expert = EnvExpert(robot_init_qpos, start_frame, end_frame, obj_type=obj_type, render=render,
                               model_xml=xml_path)
        env_expert.reset(mode="hard_ctrl")
        t, last_t = 0, 0
        last_frame = env_expert.env.sim.data.qpos[-7:].copy()
        while True:
            t += 1
            action = env_expert.get_expert_action(state=None)
            state, reward, done, success, info = env_expert.step(action)
            if slip_detection:
                cur_frame = env_expert.env.sim.data.qpos[-7:]
                # print("last frame", last_frame)
                # print("cur frame", cur_frame)
                delta_p = get_pos_err(cur_frame[:3], last_frame[:3])
                _, delta_q = get_rel_angle_axis(cur_frame[-4:], last_frame[-4:])
                # print("delta_p", delta_p)
                # print("delta_q", delta_q)
                if delta_p > dp_thresh or delta_q > dq_thresh:
                    if t - last_t < dt_thresh:
                        log("Slipped. dp={:.4f}, dq={:.4f}, dt={:.4f}".format(delta_p, delta_q, t - last_t))
                        done = True
                    else:
                        last_t = t
                        last_frame = cur_frame.copy()
            if done:
                print("Success: {} Error: {:.4f} ".format(success, env_expert.min_err))
                if success:
                    contact_points_log.append(env_expert.contact_points_seq.copy())
                break
        instance = np.hstack((start_frame, end_frame, np.array([success])))
        print(instance)
        collision.append(instance)
        if (i + 1) % 100 == 0:
            np.save(os.path.join(save_folder, data_file), np.array(collision))
            np.save(os.path.join(save_folder, "robot_init_qpos_log.npy"), np.array(robot_init_qpos_log))
            if use_contact_points_log:
                np.save(os.path.join(save_folder, "contact_points_log.npy"), contact_points_log)


def make_instance(from_frame, to_frame, reachability, rrt_result):
    """
    entry format:
    [0:7] : from_frame
    [7:14]: to_frame
    [14]  : reachability
    [15]  : rrt result (1: success, 0: timeout)
    """
    print(from_frame, to_frame, reachability, rrt_result)
    instance = np.zeros(16, dtype=np.float32)
    instance[:7] = from_frame
    instance[7:14] = to_frame
    instance[14] = reachability
    instance[15] = rrt_result
    return instance


def collision_dataset_append(collision_dataset_file, rrt_folder):
    if os.path.exists(collision_dataset_file):
        collision_dataset = np.load(collision_dataset_file)
    else:
        collision_dataset = np.zeros((1, 16))  # dummy

    df = pandas.read_excel(os.path.join(rrt_folder, "results.xlsx"), engine='openpyxl')
    for index, row in df.iterrows():
        no, success = row['no'], row['success']
        if os.path.exists(os.path.join(rrt_folder, str(no), "collision_log.npy")):
            log("loading collision log...")
            collision_log = np.load(os.path.join(rrt_folder, str(no), "collision_log.npy"))
            collision_log = np.append(collision_log, 2 * np.zeros(collision_log.shape[0], 1), axis=1)
            assert collision_log.shape[1] == 16
            collision_dataset = np.vstack((collision_dataset, collision_log))
        if success:
            path = load_path(os.path.join(rrt_folder, str(no), "path.data"))
            path_frames, beacon_frames = get_frames_from_path(path)
            n_steps = len(path_frames) - 1
            # print("n_steps:", n_steps)
            for i in range(n_steps):
                from_frame = path_frames[i]
                if i == n_steps - 1:
                    to_frame = beacon_frames[-1]
                else:
                    to_frame = path_frames[i + 1]
                instance = make_instance(from_frame, to_frame, reachability=1, rrt_result=1)
                collision_dataset = np.vstack((collision_dataset, instance.reshape(1, -1)))
            for i in range(n_steps - 1):
                instance = make_instance(path_frames[i], beacon_frames[-1], reachability=0, rrt_result=1)
                collision_dataset = np.vstack((collision_dataset, instance.reshape(1, -1)))
        else:
            instance = make_instance(np.fromstring(row['start frame'][1:-1], sep=' ', dtype=np.float32),
                                     np.fromstring(row['end frame'][1:-1], sep=' ', dtype=np.float32),
                                     reachability=0, rrt_result=0)
            collision_dataset = np.vstack((collision_dataset, instance.reshape(1, -1)))
    np.save(collision_dataset_file, collision_dataset)


def extract_success_cases(data_file="collision_dataset/cube.npy"):
    all_cases = np.load(data_file)
    success_cases = all_cases[all_cases[:, -1] == 1.]
    print("success: {} / {}".format(success_cases.shape[0], all_cases.shape[0]))
    np.save(data_file.replace(".npy", "_success.npy"), success_cases)


if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='0')
    parser.add_argument('--obj_type', type=str, default='cube')
    parser.add_argument('--obj_r', type=float, default=0.03)
    parser.add_argument('--angle_low', type=float, default=0.)
    parser.add_argument('--angle_high', type=float, default=180.)
    parser.add_argument('--n_instance', type=int, default=6000)
    parser.add_argument('--seed_low', type=int, default=0)
    parser.add_argument('--seed_high', type=int, default=int(1e6))
    parser.add_argument('--contact_points_log', action='store_true')
    parser.add_argument('--slip_detection', action='store_true')
    parser.add_argument('--sample_near_success', action='store_true')
    parser.add_argument('--success_file', type=str, default=None)
    parser.add_argument('--success_id_low', type=int, default=0)
    parser.add_argument('--success_id_high', type=int, default=1000)
    args = parser.parse_args()

    obj_type, obj_r = args.obj_type, args.obj_r
    angle_range = np.array([args.angle_low, args.angle_high])
    save_folder = "collision_data_" + obj_type + str(angle_range) + "_" + args.name

    collect_collision_data(obj_type=obj_type, obj_r=obj_r, n_instance=args.n_instance,
                           max_pos_dev=np.array([0.02, 0.02, 0.01]), angle_range=angle_range,
                           slip_detection=args.slip_detection, use_contact_points_log=args.contact_points_log,
                           data_file="collision.npy",
                           save_folder=save_folder,
                           seed_log_file=os.path.join(save_folder, "seed_log.npy"),
                           seed_range=(args.seed_low, args.seed_high),
                           sample_near_success=args.sample_near_success,
                           success_file=args.success_file,
                           success_id_low=args.success_id_low, success_id_high=args.success_id_high,
                           render=False)

    # To augment data by exploiting the rotational symmetries of the objects
    # augment_data(obj_type=obj_type,
    #              raw_data_file="collision_dataset/cube_no_aug_train.npy",
    #              augmented_data_file="collision_dataset/cube_aug_train.npy")

    # To extract success cases from a collected dataset
    # extract_success_cases(data_file="collision_dataset/cube.npy")
