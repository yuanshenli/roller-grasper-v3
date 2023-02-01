from utils import *
from config import *
from expert import get_init_contact_points
from collision_oracle_pc import CollisionOracle
from sampling import generate_random_case
import numpy as np
import pandas
import time
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)


class RRT_PC(object):
    """
    Class for RRT Expand planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, pos, quat, node_id=None, score=1.):
            """
            pos: (3,)
            quat: (4,)
            """
            self.pos = pos.copy()
            self.quat = quat.copy()
            self.beacon_pos = None
            self.beacon_quat = None
            self.parent = None
            self.node_id = node_id
            self.score = score

    def __init__(self,
                 start,
                 goal,
                 center_pos=np.array([0., 0., 0.2]),
                 max_pos_deviation=np.array([0.02, 0.02, 0.01]),
                 pos_err_thresh=0.005,
                 quat_err_thresh=0.05,
                 max_expand_p=0.07,
                 max_expand_q=120.0 * np.pi / 180.0,
                 expand_p=0.006,
                 expand_q=11.0 * np.pi / 180.0,
                 expand_t=10,
                 goal_sample_rate=20,
                 max_iter=500,
                 exp_folder="rrt_pc",
                 collision_oracle_folder="collision_oracle_pc",
                 restore_step=100000,
                 raw_point_cloud_file="point_cloud/cube.npy",
                 discard_rate=0.2,
                 blacklist_file=None,
                 blacklist_thresh=0.,
                 score_thresh=0.75,
                 log_permit=False):
        self.start = self.Node(start[0:3], start[-4:], node_id=0)
        self.end = self.Node(goal[0:3], goal[-4:], node_id=-1)
        self.end.beacon_pos = goal[0:3].copy()
        self.end.beacon_quat = goal[-4:].copy()
        self.center_pos = center_pos
        self.max_pos_deviation = max_pos_deviation
        self.pos_err_thresh = pos_err_thresh
        self.quat_err_thresh = quat_err_thresh
        self.pos_err_scale = 1 / pos_err_thresh if pos_err_thresh > 0 else 0
        self.quat_err_scale = 1 / quat_err_thresh if quat_err_thresh > 0 else 0
        self.expand_p = expand_p
        self.expand_q = expand_q
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.node_cnt = 0
        self.tree_size = 0
        self.exp_folder = exp_folder
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)
        self.max_expand_p = max_expand_p
        self.max_expand_q = max_expand_q
        self.expand_t = expand_t
        self.discard_rate = discard_rate
        self.blacklist_file = blacklist_file
        if blacklist_file is None:
            self.blacklist = None
        elif os.path.exists(blacklist_file):
            self.blacklist = np.load(blacklist_file)
        else:
            self.blacklist = None
        self.blacklist_thresh = blacklist_thresh
        self.score_thresh = score_thresh
        self.collision_oracle_folder = collision_oracle_folder
        self.collision_oracle = CollisionOracle(device=device,
                                                save_top_dir=collision_oracle_folder,
                                                raw_point_cloud_file=raw_point_cloud_file)
        self.collision_oracle.restore(restore_step)
        self.log_permit = log_permit

    def planning(self, prune=True, random_seed=0, numpy_seed=0):
        """
        rrt expand path planning
        """
        random.seed(random_seed)
        np.random.seed(numpy_seed)
        self.node_cnt = 0
        self.tree_size = 0
        self.node_list = [self.start]
        if not self.is_in_blacklist(np.hstack((self.start.pos, self.start.quat)),
                                    np.hstack((self.end.pos, self.end.quat))) and \
                self.predict_collision(self.start, self.end):
            return self.generate_final_course(len(self.node_list) - 1)

        goal_sample_prune_indices = []

        for i in range(self.max_iter):
            nearest_node_index = self.get_nearest_node_index(self.node_list, self.end, criterion="weighted")
            nearest_node = self.node_list[nearest_node_index]
            min_pos_err = get_pos_err(nearest_node.pos, self.end.pos)
            min_quat_err = get_quat_err(nearest_node.quat, self.end.quat)
            if self.log_permit:
                log("iter: {}, tree size: {}, remaining pos err: {:.4f}, quat err: {:.4f}, weighted err: {:.4f}".format(
                    i, len(self.node_list), min_pos_err, min_quat_err, self.get_weighted_err(min_pos_err, min_quat_err)
                ), 'b', 'B')

            # rnd_node = self.get_goal_node() if i == 0 else self.sample_node()
            rnd_node = self.sample_node()
            near_node_index = self.get_nearest_node_index(self.node_list, rnd_node, criterion="weighted")
            near_node = self.node_list[near_node_index]
            if self.is_in_blacklist(np.hstack((near_node.pos, near_node.quat)),
                                    np.hstack((rnd_node.pos, rnd_node.quat))):
                continue
            if self.is_same_node(rnd_node, self.end):
                if near_node_index in goal_sample_prune_indices:
                    continue  # searched before, prune
                goal_sample_prune_indices.append(near_node_index)

            for criterion in ["weighted", "quat", "pos"]:
                if prune and self.prune(near_node, rnd_node):
                    continue
                d, theta, _ = self.calc_distance_and_angle(near_node, rnd_node)
                if criterion == "pos":
                    expand_p_list = np.linspace(self.expand_p, self.max_expand_p,
                                                round(self.max_expand_p / self.expand_p))
                    expand_q_list = np.zeros_like(expand_p_list)
                    # expand_path = self.expand(near_node, rnd_node, expand_p_list, expand_q_list)
                    expand_path = self.fast_expand(near_node, rnd_node, expand_p_list, expand_q_list)
                elif criterion == "quat":
                    expand_q_list = np.linspace(self.expand_q, self.max_expand_q,
                                                round(self.max_expand_q / self.expand_q))
                    expand_p_list = np.zeros_like(expand_q_list)
                    # expand_path = self.expand(near_node, rnd_node, expand_p_list, expand_q_list)
                    expand_path = self.fast_expand(near_node, rnd_node, expand_p_list, expand_q_list)
                else:
                    if d == 0 or theta == 0:
                        continue
                    r = min(self.max_expand_p / d, self.max_expand_q / theta)
                    max_expend_p = r * d
                    max_expend_q = r * theta
                    n_nodes = max(1, round(min(max_expend_p / self.expand_p, max_expend_q / self.expand_q)))
                    expand_p_list = np.linspace(max_expend_p / n_nodes, max_expend_p, n_nodes)
                    expand_q_list = np.linspace(max_expend_q / n_nodes, max_expend_q, n_nodes)
                    # expand_path = self.expand(near_node, rnd_node, expand_p_list, expand_q_list)
                    expand_path = self.fast_expand(near_node, rnd_node, expand_p_list, expand_q_list)

                new_node_list = []
                for j, node in enumerate(expand_path):
                    if self.is_in_blacklist(np.hstack((node.parent.pos, node.parent.quat)),
                                            np.hstack((node.pos, node.quat))):
                        continue
                    new_node_list.append(node)
                    # self.node_list.append(node)
                if new_node_list != []:
                    goal_reachable, scores = self.predict_collision_batch(new_node_list, [self.end] * len(new_node_list))
                    if goal_reachable.any():
                        final_node_idx = goal_reachable.argmax()
                        self.end.score = scores[final_node_idx]
                        self.node_list.extend(new_node_list[:final_node_idx + 1])
                        return self.generate_final_course(len(self.node_list) - 1)
                    else:
                        self.node_list.extend(new_node_list)
        return None  # cannot find path

    def expand(self, near_node, rnd_node, p_list, q_list):
        expand_path = []
        for p, q in zip(p_list, q_list):
            new_node = self.steer(near_node, rnd_node, extend_length=p, extend_angle=q)
            if (abs(new_node.pos - self.center_pos) > self.max_pos_deviation).any():
                break
            if self.predict_collision(near_node, new_node):
                expand_path.append(new_node)
            else:
                break
        return expand_path

    def fast_expand(self, near_node, rnd_node, p_list, q_list):
        new_node_list = []
        for p, q in zip(p_list, q_list):
            new_node = self.steer(near_node, rnd_node, extend_length=p, extend_angle=q)
            if (abs(new_node.pos - self.center_pos) > self.max_pos_deviation).any():
                break
            new_node_list.append(new_node)
        if new_node_list == []:
            return new_node_list
        collision_free, _ = self.predict_collision_batch([near_node] * len(new_node_list), new_node_list)
        if collision_free.all():
            return new_node_list
        else:
            clip_idx = collision_free.argmin()
            return new_node_list[:clip_idx]

    def steer(self, from_node, to_node, extend_length=0.07, extend_angle=120 * np.pi / 180):
        d, theta, axis = self.calc_distance_and_angle(from_node, to_node)
        extend_r = R.from_rotvec(extend_angle * axis)
        r0 = R.from_quat(mjcquat_to_sciquat(from_node.quat))
        r = extend_r * r0
        new_pos = normalize_vec(to_node.pos - from_node.pos) * extend_length + from_node.pos
        new_quat = sciquat_to_mjcquat(r.as_quat())
        new_node = self.Node(new_pos, new_quat)
        new_node.beacon_pos = new_pos.copy()
        new_node.beacon_quat = new_quat.copy()

        if not self.is_same_node(new_node, to_node):
            self.node_cnt += 1
        new_node.node_id = self.node_cnt
        if not self.is_same_node(new_node, to_node):
            if self.log_permit:
                log("Steer Node!", "p", "B")
                log('Node id {}, {}'.format(new_node.node_id, np.hstack((new_node.pos, new_node.quat))), 'p', 'B')

        new_node.parent = from_node
        return new_node

    def prune(self, near_node, rnd_node, log_permit=False):
        if self.is_same_node(rnd_node, self.end) or np.array_equal(self.start.quat, self.end.quat):
            return False
        _, _, axis1 = self.calc_distance_and_angle(near_node, self.end)
        _, _, axis2 = self.calc_distance_and_angle(near_node, rnd_node)
        if np.dot(axis1, axis2) < 0:
            if log_permit:
                log('node rejected!(case 1)', 'p', "B")
            return True
        if np.dot(self.end.pos - near_node.pos, rnd_node.pos - near_node.pos) < 0:
            if log_permit:
                log('node rejected!(case 2)', 'p', "B")
            return True
        return False

    def set_point_cloud(self, point_cloud):
        self.collision_oracle.set_point_cloud(point_cloud)

    def predict_collision(self, from_node, to_node):
        from_pos, from_quat = from_node.pos, from_node.quat
        from_axis, from_theta = quat_to_rotvec(from_quat)
        to_pos, to_quat = to_node.pos, to_node.quat
        to_axis, to_theta = quat_to_rotvec(to_quat)
        input = np.hstack((from_pos, from_axis, from_theta, to_pos, to_axis, to_theta))
        score = self.collision_oracle.predict(input).item()
        to_node.score = score
        collision_free = score > self.score_thresh
        if score < self.blacklist_thresh:
            self.blacklist_append(np.hstack((from_pos, from_quat, to_pos, to_quat)))
        return collision_free

    def predict_collision_batch(self, from_node_list, to_node_list):
        input_batch_rotvec = np.zeros((len(to_node_list), 14))
        input_batch_quat = np.zeros((len(to_node_list), 14))
        for i, (from_node, to_node) in enumerate(zip(from_node_list, to_node_list)):
            from_pos, from_quat = from_node.pos, from_node.quat
            from_axis, from_theta = quat_to_rotvec(from_quat)
            to_pos, to_quat = to_node.pos, to_node.quat
            to_axis, to_theta = quat_to_rotvec(to_quat)
            input_batch_rotvec[i, :] = np.hstack((from_pos, from_axis, from_theta, to_pos, to_axis, to_theta))
            input_batch_quat[i, :] = np.hstack((from_pos, from_quat, to_pos, to_quat))
        scores = self.collision_oracle.predict(np.array(input_batch_rotvec))
        for i, to_node in enumerate(to_node_list):
            to_node.score = scores[i]
        collision_free = scores > self.score_thresh
        for frame_pair in input_batch_quat[scores < self.blacklist_thresh]:
            self.blacklist_append(frame_pair)   # update blacklist
            # log("{} added to blacklist".format(frame_pair))
        return collision_free, scores

    def get_path_from_root(self, node):
        path = []
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)
        path.reverse()
        return path

    def generate_final_course(self, goal_ind):
        node = self.node_list[goal_ind]
        path = self.get_path_from_root(node)
        path.append(self.end)
        return path

    def blacklist_append(self, frame_pair):
        if self.blacklist is None:
            self.blacklist = frame_pair.reshape(1, -1)
        else:
            if frame_pair not in self.blacklist:
                self.blacklist = np.vstack((self.blacklist, frame_pair))

    def save_blacklist(self):
        if self.blacklist_file is not None and self.blacklist is not None:
            np.save(self.blacklist_file, self.blacklist)

    def is_in_blacklist(self, from_frame, to_frame, pos_err_close=0.005, quat_err_close=0.05,
                        pos_err_near=0.03, quat_err_near=0.25):
        if self.blacklist is not None:
            for black_pair in self.blacklist:
                black_from, black_to = np.split(black_pair, [7])
                if (get_pos_err(black_from[:3], from_frame[:3]) > pos_err_close or
                        get_quat_err(black_from[-4:], from_frame[-4:]) > quat_err_close):
                    continue
                axis1, theta1 = get_rel_angle_axis(black_from[-4:], black_to[-4:])
                axis2, theta2 = get_rel_angle_axis(from_frame[-4:], to_frame[-4:])
                if (theta1 > 0. and theta2 > 0. and np.dot(axis1, axis2) > 0.95) or (
                        get_pos_err(black_from[:3], to_frame[:3]) < pos_err_near and
                        get_quat_err(black_from[-4:], to_frame[-4:]) < quat_err_near
                ):
                    if self.log_permit:
                        log("Path from {} to {} is in blacklist!".format(from_frame, to_frame))
                    return True
        return False

    def get_weighted_err(self, pos_err, quat_err):
        return self.pos_err_scale * pos_err + self.quat_err_scale * quat_err

    def get_nearest_node_index(self, node_list, rnd_node, criterion="weighted"):
        if criterion == "weighted":
            dlist = [self.get_weighted_err(get_pos_err(node.pos, rnd_node.pos), get_quat_err(node.quat, rnd_node.quat))
                     for node in node_list]
        elif criterion == "pos":
            dlist = [get_pos_err(node.pos, rnd_node.pos) for node in node_list]
        elif criterion == "quat":
            dlist = [get_quat_err(node.quat, rnd_node.quat) for node in node_list]
        else:
            raise ValueError
        min_ind = dlist.index(min(dlist))
        return min_ind

    def get_goal_node(self):
        goal_node = self.Node(self.end.pos, self.end.quat, self.node_cnt)
        return goal_node

    def get_random_node(self):
        rnd_pos = np.random.uniform(self.center_pos - self.max_pos_deviation, self.center_pos + self.max_pos_deviation)
        rnd_quat = R.random().as_quat()
        self.node_cnt += 1
        rnd_node = self.Node(rnd_pos, rnd_quat, self.node_cnt)
        return rnd_node

    def sample_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            node = self.get_random_node()
            if self.log_permit:
                log("Random Sample!", 'p', 'B')
        else:  # goal point sampling
            node = self.get_goal_node()
            if self.log_permit:
                log("Goal Sample!", 'p', 'B')
        if self.log_permit:
            log('Node id {}, {}'.format(node.node_id, np.hstack((node.pos, node.quat))), 'p', 'B')
        return node

    def save_tree(self, file="tree.data"):
        with open(os.path.join(self.exp_folder, file), "wb") as f:
            pickle.dump(self.node_list, f)

    def load_tree(self, file="tree.data"):
        with open(file, "rb") as f:
            self.node_list = pickle.load(f)

    def save_path(self, path, file="path.data"):
        with open(os.path.join(self.exp_folder, file), "wb") as f:
            pickle.dump(path, f)

    def sample_near_node(self, node, pos_noise=0.005, angle_noise=15):
        ref_frame = np.hstack((node.pos, node.quat))
        frame = get_random_frame(sample_near_ref_frame=True, ref_frame=ref_frame,
                                 pos_noise=pos_noise, angle_noise=angle_noise)
        return self.Node(frame[:3], frame[-4:])

    def judge_robust(self, path, n_sample=10, pos_noise=0.005, angle_noise=15, random_seed=0, numpy_seed=0):
        random.seed(random_seed)
        np.random.seed(numpy_seed)
        log("init score list {}".format([node.score for node in path][1:]))
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            random_node_list = [self.sample_near_node(from_node, pos_noise=pos_noise, angle_noise=angle_noise)
                                for _ in range(n_sample)]
            collision_free, scores = self.predict_collision_batch(random_node_list, [to_node] * n_sample)
            to_node.score = np.min(scores)
        log("robust score list {}".format([node.score for node in path][1:]))

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        d = get_pos_err(from_node.pos, to_node.pos)
        axis, theta = get_rel_angle_axis(from_node.quat, to_node.quat)
        return d, theta, axis

    @staticmethod
    def is_same_node(node1, node2):
        return np.array_equal(node1.pos, node2.pos) and np.array_equal(node1.quat, node2.quat)


def load_path(file="path.data"):
    with open(file, "rb") as f:
        path = pickle.load(f)
    return path


def blacklist_append(frame_pair, blacklist_file="blacklist.npy"):
    if not os.path.exists(blacklist_file):
        np.save(blacklist_file, frame_pair.reshape(1, -1))
    else:
        blacklist = np.load(blacklist_file)
        print("black_list:", blacklist)
        if frame_pair not in blacklist:
            np.save(blacklist_file, np.vstack((blacklist, frame_pair)))


def get_partial_point_cloud(obj_type, raw_point_cloud, start_frame, r=0.005, save=False, render=False, visual=False):

    contact_points = get_init_contact_points(start_frame, obj_type=obj_type, render=render)
    print("contact points", contact_points)

    # obj_point_cloud = np.load("point_cloud/{}_poisson_1024.npy".format(obj_type))
    redundant_points = []
    partial_point_cloud = []
    for obj_point in raw_point_cloud:
        distances = np.linalg.norm(obj_point.reshape(1, -1) - contact_points, axis=1)
        # print(distances)
        with np.errstate(invalid='ignore'):
            if (distances < r).any():
                redundant_points.append(obj_point)
            else:
                partial_point_cloud.append(obj_point)
    redundant_points = np.array(redundant_points)
    partial_point_cloud = np.array(partial_point_cloud)
    print("raw point cloud", raw_point_cloud.shape)
    print("partial point cloud", partial_point_cloud.shape)

    if save:
        os.makedirs("partial_point_cloud", exist_ok=True)
        np.save("partial_point_cloud/{}_poisson_1024_r_{}_{}pts.npy".format(obj_type, r, partial_point_cloud.shape[0]),
                partial_point_cloud)
    if visual:
        ax = plt.axes(projection="3d")
        ax.scatter3D(raw_point_cloud[:, 0], raw_point_cloud[:, 1], raw_point_cloud[:, 2],
                     s=np.ones(raw_point_cloud.shape[0]) / 2, color="black")
        if len(redundant_points) > 0:
            ax.scatter3D(redundant_points[:, 0], redundant_points[:, 1], redundant_points[:, 2])
        ax.scatter3D(partial_point_cloud[:, 0], partial_point_cloud[:, 1], partial_point_cloud[:, 2],
                     s=np.ones(partial_point_cloud.shape[0]))
        ax.scatter3D(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2], color="red")
        plt.show()

    return partial_point_cloud


def rrt_pc_apply(obj_type, start_frame, end_frame, raw_point_cloud_file, init_transl=np.zeros(3),
                 exp_folder="rrt_pc", sub_dir=None,
                 collision_oracle_folder="collision_oracle", restore_step=2000,
                 blacklist_file=None, blacklist_thresh=0.,
                 robust=False, robust_n_sample=10, robust_pos_noise=0.005, robust_angle_noise=15,
                 use_partial_point_cloud=False, r=0.010,
                 max_iter=500, pos_err_thresh=0.005, quat_err_thresh=0.05, prune=True,
                 random_seed=0, numpy_seed=0, n_trial=10, max_allowed_steps=5, min_allowed_steps=1):
    if sub_dir is None:
        sub_dir = datetime.datetime.now().strftime("%m.%d_%H:%M:%S")
    exp_folder = os.path.join(exp_folder, sub_dir)
    os.makedirs(exp_folder, exist_ok=True)
    np.save(os.path.join(exp_folder, "init_transl.npy"), init_transl)
    log('Start Frame {}, End Frame {}'.format(start_frame, end_frame), 'c', 'B')
    init_pos_err = get_pos_err(start_frame[0:3], end_frame[0:3])
    init_quat_err = get_quat_err(start_frame[-4:], end_frame[-4:])
    log('init pos err {:.5f}, quat err {:.5f}'.format(init_pos_err, init_quat_err), 'c', 'B')

    rrt_pc = RRT_PC(
        start=start_frame,
        goal=end_frame,
        center_pos=np.array([0., 0., 0.2]),
        max_pos_deviation=np.array([0.015, 0.015, 0.01]),
        pos_err_thresh=pos_err_thresh,
        quat_err_thresh=quat_err_thresh,
        expand_p=0.006,
        expand_q=11 * np.pi / 180.0,
        expand_t=10,
        goal_sample_rate=20,
        max_iter=max_iter,
        discard_rate=0.2,
        exp_folder=exp_folder,
        blacklist_file=blacklist_file,
        blacklist_thresh=blacklist_thresh,
        collision_oracle_folder=collision_oracle_folder,
        restore_step=restore_step,
        raw_point_cloud_file=raw_point_cloud_file
    )

    if use_partial_point_cloud:
        partial_point_cloud = get_partial_point_cloud(obj_type=obj_type, raw_point_cloud=np.load(raw_point_cloud_file),
                                                      start_frame=start_frame, r=r, visual=False)
        rrt_pc.set_point_cloud(partial_point_cloud)

    best_idx = None
    max_score = 0.
    best_score_list = None
    candidate_paths, all_info = [], []

    for t in range(n_trial):
        log("Trial [{}]".format(t), "b", "B")
        start_time = time.time()
        path = rrt_pc.planning(prune=prune, random_seed=random_seed + t, numpy_seed=numpy_seed + t)
        planning_time = time.time() - start_time
        rrt_pc.save_blacklist()

        steps = 0
        if path is None or len(path) - 1 > max_allowed_steps or len(path) - 1 < min_allowed_steps:
            success = False
            log("Fail! Time: {:.0f}s".format(planning_time), 'r', 'B')
        else:
            success = True
            steps = len(path) - 1
            log("Succeed! Time: {:.0f}s, Number of Steps: {}".format(planning_time, steps), 'g', 'B')
            if robust:
                rrt_pc.judge_robust(path, n_sample=robust_n_sample, pos_noise=robust_pos_noise,
                                    angle_noise=robust_angle_noise)
            for i in range(len(path)):
                print(path[i].pos, path[i].quat)
                if i + 1 < len(path):
                    d, theta, _ = rrt_pc.calc_distance_and_angle(path[i], path[i + 1])
                    print("Distance: pos {:.4f} angle {:.4f}".format(d, theta * 180 / np.pi))

            rrt_pc.save_tree(file="tree_{}.data".format(t))
            rrt_pc.save_path(file="path_{}.data".format(t), path=path)

            score = 1.
            score_list_i = []
            for node in path:
                score_list_i.append(node.score)
                # score *= node.score
            score = min(score_list_i)
            log("Score list: {}".format(np.array(score_list_i[1:])))
            log("Final score: {:.4f}".format(score))
            if score > max_score:
                max_score = score
                best_idx = t
                best_score_list = np.array(score_list_i)

        info = {'start frame': start_frame,
                'end frame': end_frame,
                'init transl': init_transl,
                'init pos err': init_pos_err,
                'init quat err': init_quat_err,
                'success': success,
                'planning time': planning_time,
                'steps': steps}
        candidate_paths.append(path)
        all_info.append(info)

    if best_idx is None:
        return None, None
    else:
        log("*" * 80, "b", "B")
        log("Start frame: {}\nEnd frame: {}".format(start_frame, end_frame), "b", "B")
        log("Best trial index: {}/{}".format(best_idx, n_trial), "b", "B")
        log("Path:", "b", "B")
        frame_list = []
        best_path = candidate_paths[best_idx]
        for i, node in enumerate(best_path):
            frame = np.hstack((node.pos, node.quat))
            frame_list.append(frame)
            log(frame, "b", "B")
        log("#Steps: {}".format(len(best_path) - 1), "b", "B")
        log("Scores: {}".format(best_score_list[1:]), "b", "B")
        best_path_file = os.path.join(exp_folder, "best_path.npy")
        np.save(best_path_file, frame_list)
        log("Best path saved to \"{}\"".format(best_path_file), "b", "B")
        log("*" * 80, "b", "B")

    return candidate_paths[best_idx], all_info[best_idx]


def get_start_end_frame_from_path(path):
    start_node, end_node = path[0], path[-1]
    start_frame = np.hstack((start_node.pos, start_node.quat))
    end_frame = np.hstack((end_node.beacon_pos, end_node.beacon_quat))
    return start_frame, end_frame


def get_frames_from_path(path):
    path_frames = []
    beacon_frames = []
    for node in path:
        path_frames.append(np.hstack((node.pos, node.quat)))
        beacon_frames.append(np.hstack((node.beacon_pos, node.beacon_quat)))
    return path_frames, beacon_frames


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="exp_rrt_pc", help="directory to save experiment output")
    parser.add_argument("--obj_type", type=str, help="object type",
                        choices=["cube", "rectangular", "cube_w_opening", "mug", "prism_w_handle"]
                        )
    parser.add_argument("--model_type", type=str, help="specify model type to be single or unified",
                        choices=["single", "unified"]
                        )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    raw_point_cloud_file = f"point_cloud_test/{args.obj_type}.npy"
    collision_oracle_folder = f"data/checkpoints/{args.obj_type}"
    if args.model_type == "unified":
        collision_oracle_folder = f"data/checkpoints/unified"
        restore_step = 700000
    elif args.obj_type == "cube":
        restore_step = 100000
    elif args.obj_type == "cube_w_opening":
        restore_step = 400000
    elif args.obj_type == "mug":
        restore_step = 590000
    elif args.obj_type == "rectangular":
        restore_step = 350000
    else:
        raise ValueError(f"no single model for object f{args.obj_type}")

    start_frame = np.array([0., 0., 0.2, 0.7071, 0., 0., 0.7071])
    end_frame = np.array([0.,    -0.0171,  0.1907, 0.2263,  0.9389, -0.2588, 0.0182])

    path, _ = rrt_pc_apply(obj_type=args.obj_type, start_frame=start_frame, end_frame=end_frame,
                           exp_folder=args.output_dir,
                           raw_point_cloud_file=raw_point_cloud_file,
                           collision_oracle_folder=collision_oracle_folder, restore_step=restore_step,
                           n_trial=10,  # pick the best out of 10
                           robust=True, robust_n_sample=10, robust_pos_noise=0.005, robust_angle_noise=15,
                           blacklist_file=None, blacklist_thresh=0.,
                           use_partial_point_cloud=False, r=0.010,
                           max_allowed_steps=8, min_allowed_steps=2,
                           numpy_seed=args.seed, random_seed=args.seed)
