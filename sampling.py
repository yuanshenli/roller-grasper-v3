from constant import *
from utils import *
from expert import get_steady_frame
import numpy as np
import pandas as pd
import pickle


def sample_transl(obj_type):
    transl = np.zeros(3)
    if obj_type in ["cube", "cube_w_opening", "prism_w_handle", "mug"]:
        pass
    elif obj_type == "rectangular2":
        rnd = np.random.uniform(-0.034, 0.034)
        transl = np.array([0., 0., rnd])
    elif obj_type == "rectangular":
        transl = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.02, 0.02), 0])
    else:
        raise ValueError("Unknown object type!")
    return transl


def generate_random_case(obj_type="cube", obj_r=0.03, frame_stability_check=True, use_transl=False,
                         init_pos=np.array([0.0, 0.0, 0.20]), max_pos_dev=np.array([0.02, 0.02, 0.02]),
                         quat_upright=False, angle_range=np.array([0, 180]), init_err_min=10,
                         sample_near_success=False, success_case=None, pos_noise=0.005, angle_noise=15,
                         random_seed=None, numpy_seed=None, seed_log_file="seed_log.npy", seed_range=(0, int(1e9))):
    # Generate Random Start Frame & End Frame
    if random_seed is None or numpy_seed is None:
        random_seed, numpy_seed = get_random_seed(seed_log_file=seed_log_file, seed_range=seed_range)
    log('random_seed {}, numpy_seed {}'.format(random_seed, numpy_seed), 'c', 'B')
    if use_transl:
        transl_start = sample_transl(obj_type)
        transl_end = sample_transl(obj_type)
    else:
        transl_start, transl_end = np.zeros(3), np.zeros(3)
    if sample_near_success:
        assert success_case is not None
        print("success case", success_case)
        while True:
            start_frame = get_random_frame(init_pos, max_pos_dev, transl_start, quat_upright=quat_upright,
                                           sample_near_ref_frame=True, ref_frame=success_case[0:7],
                                           pos_noise=pos_noise, angle_noise=angle_noise)
            end_frame = get_random_frame(init_pos, max_pos_dev, transl_start, quat_upright=quat_upright,
                                         sample_near_ref_frame=True, ref_frame=success_case[7:14],
                                         pos_noise=pos_noise, angle_noise=angle_noise)
            init_pos_err = get_pos_err(start_frame[:3], end_frame[:3])
            init_quat_err = get_quat_err(start_frame[-4:], end_frame[-4:])
            init_err = SCALE_ERROR_POS * init_pos_err + SCALE_ERROR_ROT * init_quat_err
            axis, theta = get_rel_angle_axis(start_frame[-4:], end_frame[-4:])
            angle = theta * 180 / np.pi
            if init_err >= init_err_min and angle_range[0] <= angle <= angle_range[1]:
                break
    else:
        start_frame = get_random_frame(init_pos, max_pos_dev, transl_start, quat_upright=quat_upright)
        while True:
            end_frame = get_random_frame(init_pos, max_pos_dev, transl_end, quat_upright=quat_upright)
            init_pos_err = get_pos_err(start_frame[:3], end_frame[:3])
            init_quat_err = get_quat_err(start_frame[-4:], end_frame[-4:])
            init_err = SCALE_ERROR_POS * init_pos_err + SCALE_ERROR_ROT * init_quat_err
            axis, theta = get_rel_angle_axis(start_frame[-4:], end_frame[-4:])
            angle = theta * 180 / np.pi
            if init_err >= init_err_min and angle_range[0] <= angle <= angle_range[1]:
                break
    waypoints = [start_frame, end_frame]
    transls = [transl_start, transl_end]

    # Check Frame Stability
    steady = None
    if frame_stability_check:
        steady = True
        for j, frame in enumerate(waypoints):
            steady_frame = get_steady_frame(frame, transls[j], obj_type=obj_type, obj_r=obj_r, render=False,
                                            debug=False)
            if steady_frame is None:
                log("Frame {} is unsteady!".format(j), "y", "B")
                steady = False
                break
            waypoints[j] = steady_frame.copy()
        start_frame, end_frame = waypoints[0], waypoints[1]

    info = {"steady": steady,
            "transl_start": transl_start,
            "transl_end": transl_end,
            "gen_random_seed": random_seed,
            "gen_numpy_seed": numpy_seed}

    return start_frame, end_frame, info


if __name__ == "__main__":
    start_frame, end_frame, info = generate_random_case()
    info["start_frame"] = start_frame
    info["end_frame"] = end_frame

    with open("random_case.data", "wb") as f:
        pickle.dump(info, f)

    with open("random_case.data", "rb") as f:
        got_info = pickle.load(f)
    print(got_info)

