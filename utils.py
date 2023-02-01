import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio
import os
import sys
import torch.utils.tensorboard as tb
import datetime
import random
import fcntl

import torch
from torch.optim.lr_scheduler import LambdaLR
from bisect import bisect_right

LOG_PERMIT = True


def get_pos_err(cur_pos, target_pos):
    err = np.linalg.norm(cur_pos - target_pos)
    return err


def get_quat_err(cur_quat, target_quat):
    # Mujoco quat (w,x,y,z)
    err = min(np.linalg.norm(cur_quat - target_quat), np.linalg.norm(cur_quat + target_quat))
    err /= np.sqrt(2)
    return err


def log(string, color='', style='', back=''):
    if LOG_PERMIT is False:
        return
    color_dict = {'k': ';30', 'r': ';31', 'g': ';32', 'y': ';33',
                  'b': ';34', 'p': ';35', 'c': ';36', 'w': ';37', '': ''}
    style_dict = {'N': '0', 'B': '1', 'I': '3', 'U': '4', '': '0'}
    back_dict = {'k': ';40', 'r': ';41', 'g': ';42', 'y': ';43',
                 'b': ';44', 'p': ';45', 'c': ';46', 'w': ';47', '': ''}
    ENDC = '\033[0m'
    cur_color = '\033[' + style_dict[style] + color_dict[color] + back_dict[back] + 'm'
    print(cur_color + str(string) + ENDC)


def get_random_seed(seed_log_file="seed_log.npy", seed_range=(0, int(1e6))):
    random_seed = np.random.randint(seed_range[0], seed_range[1])
    numpy_seed = np.random.randint(seed_range[0], seed_range[1])
    if not os.path.exists(seed_log_file):
        with open(seed_log_file, 'w') as f:
            # fcntl.flock(f, fcntl.LOCK_EX)
            np.save(seed_log_file, [[random_seed, numpy_seed]])
    else:
        with open(seed_log_file, 'r+') as f:
            # fcntl.flock(f, fcntl.LOCK_EX)
            seed_list = np.load(seed_log_file).tolist()
            while [random_seed, numpy_seed] in seed_list:
                random_seed = np.random.randint(seed_range[0], seed_range[1])
                numpy_seed = np.random.randint(seed_range[0], seed_range[1])
            seed_list.append([random_seed, numpy_seed])
            np.save(seed_log_file, seed_list)
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    return random_seed, numpy_seed


def quat_to_euler(quat, degrees=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler('xyz', degrees=degrees)


def euler_to_quat(euler, sequence="xyz", degrees=False):
    r = R.from_euler(sequence, [euler[0], euler[1], euler[2]], degrees=degrees)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


def quat_to_rotvec(quat, degrees=False):
    r = R.from_quat(mjcquat_to_sciquat(quat))
    rotvec = r.as_rotvec()
    if np.linalg.norm(rotvec) == 0:
        raise ValueError("Zero rotvec!")
    axis = normalize_vec(rotvec)
    theta = np.linalg.norm(rotvec)
    if degrees:
        theta = theta * 180.0 / np.pi
    return axis, theta


def rotvec_to_quat(axis, theta, degrees=False):
    if degrees:
        theta = theta * 180.0 / np.pi
    n_axis = normalize_vec(axis)
    quat = np.array([np.cos(theta / 2),
                     n_axis[0] * np.sin(theta / 2),
                     n_axis[1] * np.sin(theta / 2),
                     n_axis[2] * np.sin(theta / 2)])
    return quat


def rotvec_to_quat_batch(axis, theta, degrees=False):
    if degrees:
        theta = theta * 180.0 / np.pi
    # axis set to (0, 0, 0) if norm is 0
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    norm_divider = norm.copy()
    norm_divider[norm_divider == 0] = 1
    norm_multiplier = np.ones_like(norm)
    norm_multiplier[norm == 0] = 0
    n_axis = axis / norm_divider * norm_multiplier
    rot_vec = n_axis * theta
    sci_quat = R.from_rotvec(rot_vec).as_quat()
    mjc_quat = sciquat_to_mjcquat_batch(sci_quat)
    return mjc_quat


def euler_to_rotvec(euler, sequence="xyz", degrees=False):
    return quat_to_rotvec(euler_to_quat(euler, sequence=sequence, degrees=degrees))


def mjcquat_to_sciquat(mjc_quat):
    return np.array([mjc_quat[1], mjc_quat[2], mjc_quat[3], mjc_quat[0]])


def mjcquat_to_sciquat_batch(mjc_quat):
    if mjc_quat.ndim == 1:
        return np.array([mjc_quat[1], mjc_quat[2], mjc_quat[3], mjc_quat[0]])
    else:
        return np.hstack((mjc_quat[:, 1:2], mjc_quat[:, 2:3], mjc_quat[:, 3:4], mjc_quat[:, 0:1]))


def sciquat_to_mjcquat(sci_quat):
    return np.array([sci_quat[3], sci_quat[0], sci_quat[1], sci_quat[2]])


def sciquat_to_mjcquat_batch(sci_quat):
    if sci_quat.ndim == 1:
        return np.array([sci_quat[3], sci_quat[0], sci_quat[1], sci_quat[2]])
    else:
        return np.hstack((sci_quat[:, 3:4], sci_quat[:, 0:1], sci_quat[:, 1:2], sci_quat[:, 2:3]))


def normalize_vec(vec):
    mag_vec = np.linalg.norm(vec)
    if mag_vec == 0:
        # raise ValueError("Cannot normalize vector of length 0")
        return np.zeros_like(vec)
    else:
        vec = vec / mag_vec
    return vec


def get_angle(v1, v2):
    cos_angle = (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))).clip(-1, 1)
    my_angle = np.arccos(cos_angle)
    return my_angle


def distance_between_lines(p1, r1, p2, r2):
    # p1 and r1 define the 1st line, p2 and r2 define the 2nd line
    n_vec = normalize_vec(np.cross(r1, r2))
    dist = 0
    if n_vec is not None:
        dist = np.dot(n_vec, (p1 - p2))
    return dist


def get_rel_angle_axis(cur_quat, target_quat):
    cur_rot_matrix = R.from_quat(mjcquat_to_sciquat(cur_quat)).as_matrix()
    target_rot_matrix = R.from_quat(mjcquat_to_sciquat(target_quat)).as_matrix()
    relative_rot_matrix = np.matmul(np.transpose(cur_rot_matrix), target_rot_matrix)
    rotvec = R.from_matrix(relative_rot_matrix).as_rotvec()
    if np.linalg.norm(rotvec) == 0:
        log("Zero rotvec!", 'y', 'B')
        return np.array([0, 0, 1]), 0.0
    axis = normalize_vec(rotvec)
    axis = np.matmul(cur_rot_matrix, axis)
    theta = np.linalg.norm(rotvec)
    return axis, theta


def quat_sym_trans(quat, sym_trans_matrix):
    # axis, angle = quat_to_rotvec(quat)
    # axis = np.matmul(sym_trans_matrix, axis)
    # sym_quat = rotvec_to_quat(axis, -angle)
    sym_quat = quat.copy()
    sym_quat[2] = -sym_quat[2]
    sym_quat[3] = -sym_quat[3]
    return sym_quat


def pos_sym_trans(pos, sym_trans_matrix):
    # sym_pos = np.matmul(sym_trans_matrix, pos)
    sym_pos = pos.copy()
    sym_pos[0] = -sym_pos[0]
    return sym_pos


def get_sphere_center(points):
    a = points[:3, :] - points[1:, :]
    b = np.sum(points[:3, :] ** 2 - points[1:, :] ** 2, axis=1) * 0.5
    d = np.linalg.det(a)
    center = []
    for i in range(3):
        d_i = a.copy()
        d_i[:, i] = b
        center.append(np.linalg.det(d_i) / d)
    return np.array(center)


def separate_task(parent_task):
    """
    Description: Separate one parent task into a combination of several child tasks
                 All tasks are represented in degrees in Euler angle
    Input:
        parent_task: numpy array representing parent task in Euler angle, (3,)
    Output:
        child_tasks: numpy array representing a combination of n several child tasks, (n,3)
    """
    origin_task = parent_task.copy()
    origin_task[origin_task == 270] = -90
    n = int(sum(abs(origin_task)) / 90)
    if n == 0:
        raise ValueError("Empty task!")
    child_tasks = np.zeros((n, 3))
    step_tasks = np.zeros((n, 3))
    num_task = 0
    while sum(abs(origin_task)) > 0:
        for i in range(3):
            if origin_task[i] == 0:
                continue
            elif origin_task[i] == 180:
                child_tasks[num_task, i] = 90
                origin_task[i] = 90
                break
            else:
                child_tasks[num_task, i] = origin_task[i]
                origin_task[i] = 0
                break
        num_task += 1

    step_tasks[0] = child_tasks[0]
    for j in range(1, n):
        step_tasks[j] = step_tasks[j - 1] + child_tasks[j]

    return child_tasks, step_tasks


def homogeneous_transformation(ref_frame, abs_frame):
    """
    Description: Get homogeneous transformation matrix from absolute frame to reference frame
    Input:
        ref_frame: numpy array concatenating position and quarternion representing reference frame(7,)
        abs_frame: numpy array concatenating position and quarternion representing absolute frame(7,)
    Output:
        rel_frame: numpy array concatenating position and quarternion representing relative frame(7,)
    """
    ref_pos, ref_quat = ref_frame[0:3], ref_frame[3:]
    abs_pos, abs_quat = abs_frame[0:3], abs_frame[3:]

    R_ref = R.from_quat(mjcquat_to_sciquat(ref_quat)).as_matrix()
    R_abs = R.from_quat(mjcquat_to_sciquat(abs_quat)).as_matrix()

    T_ref = np.vstack((np.hstack((R_ref, ref_pos.reshape(-1, 1))), np.array([0, 0, 0, 1])))
    T_abs = np.vstack((np.hstack((R_abs, abs_pos.reshape(-1, 1))), np.array([0, 0, 0, 1])))

    T_rel = np.linalg.inv(T_ref) @ T_abs

    rel_pos = T_rel[0:3, 3].reshape(-1)
    rel_quat = sciquat_to_mjcquat(R.from_matrix(T_rel[0:3, 0:3]).as_quat())
    rel_frame = np.hstack((rel_pos, rel_quat))

    return rel_frame


def get_rel_rotmat(cur_quat, target_quat):
    # get the rotmat of target w.r.t. curr quat
    cur_rot_matrix = R.from_quat(mjcquat_to_sciquat(cur_quat)).as_matrix()
    target_rot_matrix = R.from_quat(mjcquat_to_sciquat(target_quat)).as_matrix()
    relative_rot_matrix = np.matmul(np.transpose(cur_rot_matrix), target_rot_matrix)

    return relative_rot_matrix


def read_target_str(target_str, degrees=False):
    """
    Description: Parse strings representing target of single policy
    Input:
        target_str: strings representing target of single policy
    Output:
        target_axis: numpy array representing target axis, (3,)
        target_theta: target rotation angle, scalar
    """
    val = target_str.split("_")
    target_axis = normalize_vec(np.array([int(val[0]), int(val[1]), int(val[2])]))
    target_theta = int(val[3])
    if not degrees:
        target_theta = target_theta * np.pi / 180.0

    return target_axis, target_theta


def encode_target_str(target_axis, target_theta, degrees=False):
    axis_str = np.array2string(target_axis.astype(np.int), separator='_')[1:-1].replace(" ", "")
    if not degrees:
        target_str = str(int(target_theta * 180.0 / np.pi))
    else:
        target_str = str(int(target_theta))

    return axis_str + "_" + target_str


def single_policy_path(target_axis, target_theta, degrees=False):
    target_str = encode_target_str(target_axis, target_theta, degrees=degrees)
    model_path = os.path.join("single_policy", target_str, "model", "GRAC")
    return model_path


def rotate_quat(init_quat, angle, axis, degrees=False):
    target_theta = angle * np.pi / 180.0 if degrees else angle
    target_axis = normalize_vec(np.array(axis))
    target_quat = np.array([np.cos(target_theta / 2),
                            target_axis[0] * np.sin(target_theta / 2),
                            target_axis[1] * np.sin(target_theta / 2),
                            target_axis[2] * np.sin(target_theta / 2)])
    r0 = R.from_quat(mjcquat_to_sciquat(init_quat))
    extend_r = R.from_quat(mjcquat_to_sciquat(target_quat))
    r = extend_r * r0
    cur_quat = sciquat_to_mjcquat(r.as_quat())
    return cur_quat


def sample_quat_upright():
    mid_waypoints = [((0.0, 0.0, 0.20), 0, (0, 0, 1)),
                     ((0.0, 0.0, 0.20), 90, (0, 0, 1)),
                     ((0.0, 0.0, 0.20), 90, (0, 0, -1)),
                     ((0.0, 0.0, 0.20), 90, (1, 0, 0)),
                     ((0.0, 0.0, 0.20), 90, (-1, 0, 0)),
                     ((0.0, 0.0, 0.20), 180, (0, 0, 1))]
    angles = [0, 90, 180, 270]
    waypoint1 = mid_waypoints[np.random.randint(6)]
    angle = angles[np.random.randint(4)]
    waypoint2 = ((0.0, 0.0, 0.20), angle, (0, 1, 0))
    waypoints = [((0.0, 0.0, 0.20), 0, (0, 0, 1)), waypoint1, waypoint2]
    waypoints = calc_frames(waypoints, degrees=True, rel_pos=False, rel_quat=True)

    return waypoints[-1][-4:]


def get_random_frame(init_pos=np.array([0., 0., 0.2]), max_pos_dev=np.array([0.02, 0.02, 0.02]),
                     transl=np.array([0., 0., 0.]), quat_upright=False,
                     sample_near_ref_frame=False, ref_frame=None, pos_noise=0.005, angle_noise=15):
    if quat_upright:
        quat = sample_quat_upright()
        # print("quat", quat)
        rot_mat = R.from_quat(mjcquat_to_sciquat(quat)).as_matrix()
    elif sample_near_ref_frame:
        ref_quat = ref_frame[-4:]
        assert abs(np.linalg.norm(ref_quat) - 1) < 1e-5
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        quat = rotate_quat(ref_quat, angle=np.random.uniform(-angle_noise, angle_noise), axis=axis, degrees=True)
        rot_mat = R.from_quat(mjcquat_to_sciquat(quat)).as_matrix()
    else:
        rand_rot = R.random()
        rot_mat = rand_rot.as_matrix()
        quat = sciquat_to_mjcquat(rand_rot.as_quat())


    pos_offset = np.random.uniform(-max_pos_dev, max_pos_dev)
    # pos_offset = np.zeros(3)

    T01 = np.eye(4)
    T01[:3, :3] = rot_mat
    T01[:3, 3] = init_pos + pos_offset

    T12 = np.eye(4)
    T12[:3, 3] = transl

    T02 = T01 @ T12
    pos = T02[:3, 3]
    # pos = pos_offset + init_pos + rot_mat @ transl

    if sample_near_ref_frame:
        pos = ref_frame[:3] + np.random.uniform(-pos_noise, pos_noise, (3,))

    return np.hstack((pos, quat))


def calc_single_frame(waypoint, degrees=False):
    pos, angle, axis = waypoint
    target_pos = np.array(pos).copy()
    target_theta = angle * np.pi / 180.0 if degrees else angle
    target_axis = normalize_vec(np.array(axis))
    target_quat = np.array([np.cos(target_theta / 2),
                            target_axis[0] * np.sin(target_theta / 2),
                            target_axis[1] * np.sin(target_theta / 2),
                            target_axis[2] * np.sin(target_theta / 2)])
    target_frame = np.hstack((target_pos, target_quat))
    return target_frame


def calc_frames(waypoints, degrees=False, rel_pos=True, rel_quat=True):
    frames = [calc_single_frame(waypoints[i], degrees=degrees) for i in range(len(waypoints))]
    frames = np.array(frames)

    for i in range(1, len(waypoints)):
        if rel_pos:
            frames[i][0:3] = frames[i][0:3] + frames[i - 1][0:3]
        if rel_quat:
            frames[i][-4:] = rotate_quat(frames[i - 1][-4:], waypoints[i][1], waypoints[i][2], degrees=degrees)

    return frames


def save_video(queue, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    while True:
        frame = queue.get()
        if frame is None:
            break
        writer.append_data(frame)
    writer.close()


def generate_files(path, exp_folder="export"):
    # export
    if not os.path.exists(exp_folder): os.mkdir(exp_folder)

    # save
    path = os.path.join(exp_folder, path)
    if not os.path.exists(path): os.mkdir(path)

    # log
    log_folder = os.path.join(path, "log")
    logger = tb.SummaryWriter(log_folder)

    # save model
    model_folder = os.path.join(path, "model")
    if not os.path.exists(model_folder): os.mkdir(model_folder)
    model_file = os.path.join(model_folder, "GRAC")

    # save param
    param_folder = os.path.join(path, "param")
    if not os.path.exists(param_folder): os.mkdir(param_folder)
    args_txt = os.path.join(param_folder, "args.txt")

    # save result
    result_folder = os.path.join(path, "result")
    if not os.path.exists(result_folder): os.mkdir(result_folder)
    result_file = os.path.join(result_folder, "result.npy")

    # video
    video_folder = os.path.join(path, "video")
    if not os.path.exists(video_folder): os.mkdir(video_folder)
    video = os.path.join(video_folder, "video.mp4")

    return logger, model_file, args_txt, result_file, video


def save_args(path):
    """
    Save Parameters
    """
    save_vars = ['robot_init_qpos', 'obj_init_frame', 'obj_target_frame']
    save_dict = {}
    for v_name in save_vars:
        save_dict[v_name] = globals()[v_name]
        if type(save_dict[v_name]) == np.ndarray:
            save_dict[v_name] = save_dict[v_name].tolist()
    with open(path, 'w') as f:
        f.write(str(save_dict))
    f.close()


def load_args(path=None, exp_folder="export"):
    """
    Load Parameters
    """
    if path is None:
        log('Type the folder name that the parameters will be loaded from: ', 'c', 'B')
        comment = input()
        while comment not in os.listdir("export"):
            log('No such folder to load! Please enter again', 'r', 'B')
            comment = input()

        log('Load parameters from \'export/' + comment + '\'', 'g', 'B')
        path = os.path.join(exp_folder, comment, "param", 'args.txt')
    with open(path, 'r') as f:
        load_dict = eval(f.readline())
        f.close()
    load_variables = list(load_dict.keys())
    for v_name in load_variables:
        if type(load_dict[v_name]) == list:
            load_dict[v_name] = np.array(load_dict[v_name])
        globals()[v_name] = load_dict[v_name]


#######

def dict2cuda(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2cuda(v)
        elif isinstance(v, torch.Tensor):
            v = v.cuda()
        new_dic[k] = v
    return new_dic

def dict2dev(data: dict, dev):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2dev(v)
        elif isinstance(v, torch.Tensor):
            v = v.to(dev)
        new_dic[k] = v
    return new_dic

def dict2numpy(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2numpy(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().copy()
        new_dic[k] = v
    return new_dic

def dict2float(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2float(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        new_dic[k] = v
    return new_dic

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_schedule_with_warmup(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1,):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            alpha = float(current_step) / warmup_iters
            current_factor = warmup_factor * (1. - alpha) + alpha
        else:
            current_factor = 1.

        return max(0.0,  current_factor * (gamma ** bisect_right(milestones, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def add_summary(data_items: list, logger, step: int, flag: str, max_disp=4):
    for data_item in data_items:
        tags = data_item['tags']
        vals = data_item['vals']
        dtype = data_item['type']
        if dtype == 'points':
            for i in range(min(max_disp, len(tags))):
                logger.add_mesh('{}/{}'.format(flag, tags[i]),
                                vertices=vals[0], colors=vals[1], global_step=step)
        elif dtype == 'scalars':
            for tag, val in zip(tags, vals):
                if val == 'None':
                    val = 0
                logger.add_scalar('{}/{}'.format(flag, tag),
                                  val, global_step=step)
        else:
            raise NotImplementedError

class DictAverageMeter(object):
    def __init__(self):
        self.data = {}

    def update(self, new_input: dict):
        for k, v in new_input.items():
            if isinstance(v, list):
                self.data[k] = self.data.get(k, []) + v
            else:
                assert (isinstance(v, float) or isinstance(v, int)), type(v)
                self.data[k] = self.data.get(k, []) + [v]

    def mean(self):
        ret = {}
        for k, v in self.data.items():
            if not v:
                ret[k] = 'None'
            else:
                ret[k] = np.round(np.mean(v), 4)
        return ret

    def reset(self):
        self.data = {}

def calc_stat(sample, prob, scores):
    T2L = lambda x: x.float().detach().cpu().numpy().tolist()

    labels = sample['label']
    max_probs, preds = torch.max(prob, dim=1, keepdim=False)

    all_acc = torch.mean((preds == labels).float()).item()
    scores.update({'all_acc': all_acc})

    pst_inds = (preds == 1)
    if torch.sum(pst_inds) > 0:
        precision = T2L(preds[pst_inds] == labels[pst_inds])
    else:
        precision = []

    scores.update({'precision': precision})

    for thresh in [0.1, 0.25, 0.4]:
        inds = torch.abs(max_probs-0.5) > thresh
        ratio = torch.mean(inds.float()).item()
        if ratio > 0:
            th_acc = T2L(preds[inds] == labels[inds])
        else:
            th_acc = []

        scores.update({'P{}_ratio'.format(thresh): ratio,
                       'P{}_acc'.format(thresh): th_acc,
                       })


if __name__ == '__main__':
    # quat = np.array([0, 0, 0, 1])
    # euler = quat_to_euler(quat)
    # rotvec = quat_to_rotvec(quat)
    # matrix = R.from_quat(mjcquat_to_sciquat(quat)).as_matrix()
    # log("quat: {}".format(quat), 'c', 'B')
    # log("euler: {}".format(euler), 'c', 'B')
    # log("rotvec: {}".format(rotvec), 'c', 'B')
    # log("matrix:\n {}".format(matrix), 'c', 'B')
    # print()
    #
    # # Generate Random Quaternion
    # rand_euler = R.random(random_state=0).as_euler('xyz', degrees=False)
    # rand_quat = euler_to_quat(rand_euler)
    # rand_rotvec = euler_to_rotvec(rand_euler)
    # rand_matrix = R.from_quat(mjcquat_to_sciquat(rand_quat)).as_matrix()
    # log("Random quat: {}".format(rand_quat), 'g', 'B')
    # log("Random euler: {}".format(rand_euler), 'g', 'B')
    # log("Random rotvec: {}".format(rand_rotvec), 'g', 'B')
    # log("Random matrix:\n {}".format(rand_matrix), 'g', 'B')
    # print()
    #
    # # Quaternion Multiplication
    # quat = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    # r0 = R.from_quat(mjcquat_to_sciquat(quat))
    # extend_r = r0
    # r = extend_r * r0
    # log("r0: {}".format(r0.as_euler("xyz")), "b", "B")
    # log("extend: {}".format(extend_r.as_euler("xyz")), "b", "B")
    # log("new node: {}".format(r.as_euler("xyz")), "b", "B")
    # log("")
    #
    # # Verity Quaternion Error
    # extend_angle = 20.0 * np.pi / 180.0
    # axes = [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0]),
    #         normalize_vec(np.array([1, 1, 0])), normalize_vec(np.array([1, 0, 1]))]
    # for axis in axes:
    #     extend_r = R.from_rotvec(extend_angle * axis)
    #     r0 = R.from_quat(mjcquat_to_sciquat(quat))
    #     r = extend_r * r0
    #     new_quat = sciquat_to_mjcquat(r.as_quat())
    #     log("error in quat: {}".format(get_quat_err(new_quat, quat)), "y", "B")
    #     _, theta = get_rel_angle_axis(quat, new_quat)
    #     log("error in angle: {}".format(theta), "y", "B")
    #
    # # Test Rotvec
    # quat = np.array([0, 0, 0, 1])
    # quat1 = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    # quat2 = np.array([-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    # log("")
    # log("error: {}".format(get_quat_err(quat1, quat)), "c", "B")
    # log("error: {}".format(get_quat_err(quat2, quat)), "c", "B")
    # axis1, theta1 = get_rel_angle_axis(quat, quat1)
    # axis2, theta2 = get_rel_angle_axis(quat, quat2)
    # log([axis1, theta1], "c", "B")
    # log([axis2, theta2], "c", "B")
    #
    # # Read Target
    # axis, target = read_target_str("-1_-1_-1_90")
    # print(axis, target)
    # print(encode_target_str(axis, target))
    #
    # # Rotate
    # quat_1 = rotate_quat(np.array([1, 0, 0, 0]), -45, (0, 0, 1), degrees=True)
    # quat_2 = rotate_quat(quat_1, 90, (1, 0, 0), degrees=True)
    # quat_3 = rotate_quat(quat_2, -90, (0, 0, 1), degrees=True)
    # rotvec = quat_to_rotvec(quat_3, degrees=True)
    # quat_check = rotate_quat(np.array([1, 0, 0, 0]), rotvec[1], rotvec[0], degrees=True)
    # print(quat_3)
    # print(rotvec)
    # print(quat_check)

    # cr = R.from_rotvec(np.pi / 6 * np.array([1, 1, 1]))
    # tr = R.from_rotvec(np.pi / 2 * np.array([0, 1, 1]))
    # cquat = cr.as_quat()
    # tquat = tr.as_quat()
    #
    # axis, theta = get_rel_angle_axis(sciquat_to_mjcquat(cquat), sciquat_to_mjcquat(tquat))
    # print(axis, theta)

    # [-0.16173353  0.88931191  0.42774594]
    # 1.535042302866833
    ref_frame = get_random_frame(init_pos=np.array([0., 0., 0.2]), max_pos_dev=np.array([0.02, 0.02, 0.02]),
                     transl=np.array([0., 0., 0.]), quat_upright=False, sample_near_ref_frame=False)
    random_frame = get_random_frame(init_pos=np.array([0., 0., 0.2]), max_pos_dev=np.array([0.02, 0.02, 0.02]),
                     transl=np.array([0., 0., 0.]), quat_upright=False,
                     sample_near_ref_frame=True, ref_frame=ref_frame, pos_noise=0.002, angle_noise=1)
    print(ref_frame)
    print(random_frame)
    print("delta pos: {}".format(random_frame[:3] - ref_frame[:3]))
    axis, theta = get_rel_angle_axis(ref_frame[-4:], random_frame[-4:])
    print("axis: {}, angle: {}".format(axis, theta * 180 / np.pi))