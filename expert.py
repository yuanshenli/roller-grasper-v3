import numpy as np

np.set_printoptions(precision=4, suppress=False)
from robot_env import *
from config import *
from constant import *
from utils import *
from heuristic import *


class EnvExpert:
    def __init__(self, robot_pos, start_frame, end_frame, init_ctrl=None, obj_type=OBJ_TYPE, obj_fixed=False,
                 render=False, record=False, model_xml=None):
        self.obj_fixed = obj_fixed
        self.robot_pos = robot_pos.copy()
        self.start_frame = start_frame.copy()
        self.target_frame = end_frame.copy()
        self.init_qpos = np.hstack((self.robot_pos, self.start_frame))
        if init_ctrl is not None:
            self.init_ctrl = init_ctrl.copy()
        else:
            self.init_ctrl = self.robot_pos * GEAR_RATIO

        # env
        self.env = RobotEnv(self.init_qpos, self.init_ctrl, obj_fixed=obj_fixed, obj_type=obj_type,
                            render=render, record=record, model_xml=model_xml)
        self.min_err = 10000.0
        self.termination = False
        self.success = False
        self.success_thresh = 5
        self.prev_obs = self.get_obs()

        # state
        self.state = np.hstack((self.get_obs(), self.get_obs()))

        # target
        self.target_pos = end_frame[0:3].copy()
        self.target_quat = end_frame[-4:].copy()

        # error
        self.init_pos_err = get_pos_err(self.env.sim.data.qpos[-7:-4], self.target_pos)
        self.init_quat_err = get_quat_err(self.env.sim.data.qpos[-4:], self.target_quat)
        self.init_err = SCALE_ERROR_POS * self.init_pos_err + SCALE_ERROR_ROT * self.init_quat_err

        self.prev_pos_err = self.init_pos_err.copy()
        self.prev_quat_err = self.init_quat_err.copy()

        # contact
        self.contact_points_seq = []
        self.effective_ncon = self.get_effective_ncon()
        self.bad_ncon_count = 0

        # reward
        self.reward = 0

    def compute_reward(self, pos_err, quat_err, residual_action, action_scale=0.0005, step_pos_scale=10,
                       total_pos_scale=4, step_total_scale=6, ncon_scale=0.05):
        # improvement w.r.t previous frame
        pos_step_improve = self.prev_pos_err - pos_err
        quat_step_improve = self.prev_quat_err - quat_err
        step_improve = pos_step_improve * step_pos_scale + quat_step_improve

        # improvement w.r.t initial state
        pos_total_improve = np.exp(-0.1 * pos_err) - np.exp(-0.1 * self.init_pos_err)
        quat_total_improve = np.exp(-0.1 * quat_err) - np.exp(-0.1 * self.init_quat_err)
        total_improve = pos_total_improve * total_pos_scale + quat_total_improve

        # penalty w.r.t contact points
        self.effective_ncon = self.get_effective_ncon()
        if self.effective_ncon < 4:
            self.bad_ncon_count += 1
            ncon_penalty = True
        else:
            self.bad_ncon_count = 0
            ncon_penalty = False

        # penalty w.r.t residual action
        residual_penalty = np.linalg.norm(residual_action)

        # reward & penalty
        reward = step_improve * step_total_scale + total_improve - ncon_penalty * ncon_scale - \
                 residual_penalty * action_scale

        return reward

    def calibrate_loop(self, init_frame, target_frame=None, timeout=float("inf"), stage_1=True, stage_2=True):
        if self.env.viewer is not None:
            self.env.viewer._run_speed = SLOW_RUN_SPEED

        # Slightly reduce initial noise (Stage 1)
        if stage_1:
            self.wait_for_steady()
            calib_1_success = self.force_calibrate(target_pos=init_frame[0:3],
                                                   target_quat=init_frame[3:7],
                                                   ctrl_obj=True, timeout=timeout)
            if not calib_1_success: return False

        # Let pivot ready (Stage 2)
        if stage_2:
            self.wait_for_steady()
            calib_2_success = self.force_calibrate(target_pos=target_frame[0:3],
                                                   target_quat=target_frame[3:7],
                                                   ctrl_obj=False)
            if not calib_2_success: return False

        if self.env.viewer is not None:
            self.env.viewer._run_speed = RUN_SPEED

        return True

    def dummy_loop(self, render=False, debug=False):
        if render:
            self.env.render_on()
            if debug:
                self.env.viewer._paused = True
        while True:
            action = self.get_expert_action(state=None)
            state, reward, done, _, info = self.step(action, mode="RPL", log_permit=False)
            # break
            if done:
                print("Dummy Success: {}, Error: {:.4f} ".format(self.success, self.min_err))
                break

    def force_calibrate(self, target_pos, target_quat, ctrl_obj=True, timeout=float("inf")):
        pivot_thresh = 3 * np.pi / 180

        pivot_ready = False
        obj_ready = False

        while True:
            cur_qpos = self.env.sim.data.qpos.copy()
            target_pivot_qpos = []

            virtual_obj_pos = get_virtual_obj_pos(self.env.hand, self.env.sim.data.qpos)

            for i in range(4):
                cur_finger = self.env.hand[i]
                cur_finger.calibrate(observation=cur_qpos,
                                     target_pos=target_pos,
                                     target_quat=target_quat,
                                     virtual_obj_pos=virtual_obj_pos,
                                     vc_v_scale=0.0012,
                                     vc_w_scale=0.0020)
                prev_pivot_angle = self.env.sim.data.qpos[3 * i + 1]
                if not ctrl_obj:
                    delta_pivot_angle = np.clip(0 - prev_pivot_angle, -pivot_thresh, pivot_thresh)
                    target_pivot_qpos.append(0)
                else:
                    delta_pivot_angle = np.clip(cur_finger.pivot_angle - prev_pivot_angle, -pivot_thresh, pivot_thresh)
                    target_pivot_qpos.append(cur_finger.pivot_angle)

                self.env.sim.data.ctrl[3 * i + 1] += delta_pivot_angle * GEAR_RATIO[3 * i + 1]  # pivot

                if ctrl_obj:
                    self.env.sim.data.ctrl[3 * i] += cur_finger.delta_base * GEAR_RATIO[3 * i]  # base
                    self.env.sim.data.ctrl[3 * i + 2] += cur_finger.delta_roller * GEAR_RATIO[3 * i + 2]  # roller

            # Check stability
            pivot_error = np.linalg.norm(np.array(target_pivot_qpos) - cur_qpos[[1, 4, 7, 10]])
            obj_pos_error = np.linalg.norm(cur_qpos[-7:-4] - target_pos)
            obj_quat_error = get_quat_err(cur_qpos[-4:], target_quat)

            if ctrl_obj or pivot_error < 0.05: pivot_ready = True
            if not ctrl_obj or (obj_pos_error < 0.005 and obj_quat_error < 0.05): obj_ready = True
            if pivot_ready and obj_ready:
                # log("Calibration Succeeded!", "c", "B")
                self.timestep = 0
                return True
            if obj_pos_error > 0.05 or self.timestep > timeout:
                # log("Calibration Failed!", "r", "B")
                self.timestep = 0
                return False

            self.env.sim.step()
            self.timestep += 1
            if self.env.viewer is not None:
                # self.viewer.add_overlay(const.GRID_TOPRIGHT, " ", self.session_name)
                self.env.viewer.render()

    def is_drop(self, log_permit=False):
        return self.env.is_drop(log_permit=log_permit)

    def get_roller_contact(self):
        return self.env.get_roller_contact()

    def get_roller_contact_points(self):
        return self.env.get_roller_contact_points()

    def get_effective_ncon(self):
        effective_ncon = self.get_roller_contact().sum()
        return effective_ncon

    def get_obs(self, noise=False, contact=True):
        cur_obs = self.env.get_observation(contact=contact)
        if noise:
            cur_obs[-7:-4] += np.random.normal(0, 0.001, size=(3,))
            # cur_obs[-4:] += np.random.normal(0, 0.01, size=(4,))
            # new_norm = np.linalg.norm(np.array(cur_obs[-4:]))
            # cur_obs[-4:] /= new_norm
        return cur_obs

    def get_state(self, noise=False, roller_rel=True):
        curr_obs = self.get_obs(noise=noise)
        delta_obs = curr_obs - self.prev_obs
        roller_inds = [2, 5, 8, 11]
        if roller_rel:
            state = np.hstack((np.delete(curr_obs, roller_inds),
                               np.delete(self.prev_obs, roller_inds),
                               delta_obs[roller_inds]))
        else:
            state = np.hstack((curr_obs, self.prev_obs))
        # print("state dim", len(state))
        # print(state)
        return state

    def get_extend_state(self, noise=False):
        extend_state = np.hstack((self.get_state(noise=noise),
                                  self.start_frame,
                                  self.target_frame))
        return extend_state

    def get_hardctrl_action(self, target_frame, contact_correction=True, noise=False):
        cur_qpos = self.get_obs(noise=noise)
        # action = self.env.sim.data.ctrl[0:12]
        action = np.zeros(12)

        virtual_obj_pos = get_virtual_obj_pos(self.env.hand, self.env.sim.data.qpos)
        # print(cur_qpos[-7:-4], virtual_obj_pos)

        for i in range(4):
            cur_finger = self.env.hand[i]
            cur_finger.calibrate(observation=cur_qpos,
                                 target_pos=target_frame[0:3],
                                 target_quat=target_frame[3:7],
                                 virtual_obj_pos=virtual_obj_pos,
                                 vc_v_scale=0.012,
                                 vc_w_scale=0.020)

            prev_pivot_angle = self.env.sim.data.qpos[3 * i + 1]
            delta_pivot_angle = np.clip(cur_finger.pivot_angle - prev_pivot_angle, -3 * np.pi / 180, 3 * np.pi / 180)

            action[3 * i] = cur_finger.delta_base * GEAR_RATIO[3 * i]  # base
            action[3 * i + 1] = delta_pivot_angle * GEAR_RATIO[3 * i + 1]  # pivot
            action[3 * i + 2] = cur_finger.delta_roller * GEAR_RATIO[3 * i + 2]  # roller

        if contact_correction:
            roller_contact = self.get_roller_contact()
            for i in range(3):
                if roller_contact[i] == 0:
                    action[3 * i] = CONTACT_CORRECTION_ACTION * GEAR_RATIO[3 * i]

        return action

    def reset(self, robot_pos=None, start_frame=None, end_frame=None, init_ctrl=None, wait_steps=500,
              mode="hard_ctrl", noise=False):
        if robot_pos is not None:
            self.robot_pos = robot_pos.copy()
        if start_frame is not None:
            if self.obj_fixed:
                log("Invalid resetting of start_frame when object is fixed!", 'y')
            else:
                self.start_frame = start_frame.copy()
        if end_frame is not None:
            if self.obj_fixed:
                log("Invalid resetting of end_frame when object is fixed!", 'y')
            else:
                self.target_frame = end_frame.copy()
        if init_ctrl is not None:
            self.init_ctrl = init_ctrl.copy()
        self.init_qpos = np.hstack((self.robot_pos, self.start_frame))
        self.min_err = 10000.0
        self.termination = False
        self.success = False
        self.env.reset(init_qpos=self.init_qpos, init_ctrl=self.init_ctrl)
        if wait_steps > 0:
            self.wait_for_steady(wait_steps)
        self.prev_obs = self.get_obs()

        # error
        if self.obj_fixed:
            self.init_pos_err = get_pos_err(self.start_frame[:3], self.target_pos)
            self.init_quat_err = get_quat_err(self.start_frame[-4:], self.target_quat)
        else:
            self.init_pos_err = get_pos_err(self.env.sim.data.qpos[-7:-4], self.target_pos)
            self.init_quat_err = get_quat_err(self.env.sim.data.qpos[-4:], self.target_quat)
        self.init_err = SCALE_ERROR_POS * self.init_pos_err + SCALE_ERROR_ROT * self.init_quat_err

        self.prev_pos_err = self.init_pos_err.copy()
        self.prev_quat_err = self.init_quat_err.copy()

        # contact
        self.contact_points_seq = []
        self.effective_ncon = self.get_effective_ncon()
        self.bad_ncon_count = 0

        # reward
        self.reward = 0

        # initial state
        if mode == "hard_ctrl":
            state = self.get_obs(noise=noise)  # 19-d
        elif mode == "RPL":
            state = self.get_state(noise=noise)  # 38-d
        elif mode == "multi_policy":
            state = self.get_extend_state(noise=noise)  # 44-d
        else:
            raise ValueError

        return state

    def mapping(self, action):
        action_map = action.copy()
        action_map = ((MAX_REL_ACTION - MIN_REL_ACTION) / 2 * action_map + (
                MAX_REL_ACTION + MIN_REL_ACTION) / 2) * GEAR_RATIO
        action_map += self.get_hardctrl_action(self.target_frame)
        return action_map

    def standardize(self, action):
        action_st = action.copy()
        action_st = action_st - self.get_hardctrl_action(self.target_frame)
        action_st = action_st / GEAR_RATIO - (MAX_REL_ACTION + MIN_REL_ACTION) / 2
        action_st = action_st / ((MAX_REL_ACTION - MIN_REL_ACTION) / 2)
        return action_st

    def step(self, action, mode="hard_ctrl", noise=False, relative=np.tile(np.array([True, True, True]), 4),
             success_ratio=None, max_timesteps=MAX_ENV_STEPS, overlay=None,
             detach_timeout=int(1e8), pos_err_thresh=0.005, quat_err_thresh=0.05, pos_dev_limit=None,
             action_scale=0.0005, step_pos_scale=10, total_pos_scale=4, step_total_scale=6, ncon_scale=0.05,
             log_permit=False):
        # observation before step
        self.prev_obs = self.get_obs()  # 19-d
        std_residual_action = self.standardize(action)

        # step
        curr_obs = self.env.step(action, relative=relative, overlay=overlay)  # 19-d
        curr_pos = curr_obs[-7:-4]
        curr_quat = curr_obs[-4:]
        curr_pos_err = get_pos_err(curr_pos, self.target_frame[0:3])
        curr_quat_err = get_quat_err(curr_quat, self.target_frame[-4:])
        err_curr = SCALE_ERROR_POS * curr_pos_err + SCALE_ERROR_ROT * curr_quat_err

        if log_permit:
            log("Error - Pos: {} Quat: {} Timestep: {} Detach Duration: {}".format(
                curr_pos_err, curr_quat_err, self.env.timestep, self.bad_ncon_count), "p", "B")

        # reward & penalty
        mid_reward = self.compute_reward(curr_pos_err, curr_quat_err, std_residual_action, action_scale=action_scale,
                                         step_pos_scale=step_pos_scale, total_pos_scale=total_pos_scale,
                                         step_total_scale=step_total_scale, ncon_scale=ncon_scale)

        if self.min_err > err_curr:
            self.min_err = err_curr

        # termination
        if success_ratio is not None:
            pos_err_thresh = min(pos_err_thresh, success_ratio * self.init_pos_err)
            quat_err_thresh = min(quat_err_thresh, success_ratio * self.init_quat_err)
        if curr_pos_err < pos_err_thresh and curr_quat_err < quat_err_thresh:
            self.reward, self.termination, self.success, case = 0.1, True, True, 0
        elif self.is_drop() or self.bad_ncon_count > detach_timeout or \
                (pos_dev_limit is not None and curr_pos_err > pos_dev_limit):
            self.reward, self.termination, self.success, case = -0.1, True, False, 1
        elif self.env.timestep >= max_timesteps:
            self.reward, self.termination, self.success, case = mid_reward, True, False, 2
        else:
            self.reward, self.termination, self.success, case = mid_reward, False, False, 3

        if log_permit and self.termination and not self.success:
            log("Fail in case {}".format(case), "r", "B")

        # update
        self.contact_points_seq.append(self.get_roller_contact_points())
        if mode == "hard_ctrl":
            next_state = self.get_obs(noise=noise)  # 19-d
        elif mode == "RPL":
            next_state = self.get_state(noise=noise)  # 42-d
        elif mode == "multi_policy":
            next_state = self.get_extend_state(noise=noise)  # 56-d
        else:
            raise ValueError
        self.prev_pos_err = curr_pos_err.copy()
        self.prev_quat_err = curr_quat_err.copy()

        info = {"err": err_curr,
                "pos_err": curr_pos_err,
                "quat_err": curr_quat_err,
                "termination_case": case}

        return next_state, self.reward, self.termination, self.success, info

    def get_expert_action(self, state, mode="hard_ctrl", policy=None, test=False, noise=False):
        if mode == "hard_ctrl":
            action = self.get_hardctrl_action(self.target_frame, noise=noise)
        elif mode == "RPL":
            action = policy.select_action(state, test=test)
            action = self.mapping(action)
        elif mode == "multi_policy":
            action = policy.choose_action(state).detach().numpy()
            action = self.mapping(action)
        else:
            raise ValueError
        return action[0:12]

    def wait_for_steady(self, steps=10):
        prev_qpos = 1e8 * np.ones_like(self.env.sim.data.qpos)

        while True:
            cur_qpos = self.env.sim.data.qpos.copy()
            if np.linalg.norm(cur_qpos - prev_qpos) < 0.1:
                # log("Reach Steady State!", "c", "B")
                break
            self.env.sim.step()
            if self.env.viewer is not None:
                # self.viewer.add_overlay(const.GRID_TOPRIGHT, " ", self.session_name)
                self.env.viewer.render()
            prev_qpos = cur_qpos.copy()

        for _ in range(steps):
            self.env.sim.step()
            if self.env.viewer is not None:
                # self.viewer.add_overlay(const.GRID_TOPRIGHT, " ", self.session_name)
                self.env.viewer.render()

        return not self.is_drop() and self.get_effective_ncon() >= 3


def get_steady_frame(frame, transl=np.zeros(3), grasp_near=False,
                     obj_type=OBJ_TYPE, obj_r=0.03, wait_steps=500, render=False, debug=False):
    robot_init_qpos = calc_robot_init_qpos(obj_frame=frame, obj_type=obj_type, obj_r=obj_r, transl=transl, grasp_near=grasp_near)
    env_expert = EnvExpert(robot_init_qpos, frame, frame, render=render, obj_type=obj_type)
    if not env_expert.wait_for_steady(steps=wait_steps):
        return None
    if debug:
        env_expert.dummy_loop(render=True, debug=True)
    steady_frame = env_expert.env.sim.data.qpos[-7:].copy()
    return steady_frame


def get_robot_init_qpos(start_frame, obj_type=OBJ_TYPE, xml_path=None, render=False):
    robot_init_qpos = np.zeros(12)
    robot_init_qpos[[0, 3, 6]] = np.pi / 2
    robot_init_qpos[9] = -0.02
    env_expert = EnvExpert(robot_init_qpos, start_frame, start_frame,
                           obj_fixed=True, render=render, obj_type=obj_type, model_xml=xml_path)
    for t in range(200):
        action = np.zeros(12)
        action[[0, 3, 6]] = MIN_ABS_BASE * FINGER_BASE_GEAR_RATIO
        action[9] = MAX_ABS_PALM * PALM_BASE_GEAR_RATIO
        env_expert.step(action, relative=np.tile(np.array([False, False, False]), 4))
        if env_expert.get_effective_ncon() == 4:
            break
    if render:
        env_expert.env.render_off()
    robot_init_qpos = env_expert.env.sim.data.qpos[:12].copy()
    robot_init_qpos[[0, 3, 6]] -= 6 * np.pi / 180
    robot_init_qpos[9] += 0.001
    return robot_init_qpos


def get_init_contact_points(start_frame, obj_type=OBJ_TYPE, xml_path=None, render=False):
    robot_init_qpos = np.zeros(12)
    robot_init_qpos[[0, 3, 6]] = np.pi / 2
    robot_init_qpos[9] = -0.02
    env_expert = EnvExpert(robot_init_qpos, start_frame, start_frame,
                           obj_fixed=True, render=render, obj_type=obj_type, model_xml=xml_path)
    for t in range(200):
        action = np.zeros(12)
        action[[0, 3, 6]] = MIN_ABS_BASE * FINGER_BASE_GEAR_RATIO
        action[9] = MAX_ABS_PALM * PALM_BASE_GEAR_RATIO
        env_expert.step(action, relative=np.tile(np.array([False, False, False]), 4))
        if env_expert.get_effective_ncon() == 4:
            break
    if render:
        env_expert.env.render_off()

    return env_expert.get_roller_contact_points()


if __name__ == '__main__':
    obj_type = "cube"
    start_frame = calc_single_frame(((0.0, 0.0, 0.20), 0, (0, 0, 1)), degrees=True)
    end_frame = calc_single_frame(((0.0, 0.0, 0.20), 135, (0, 0, 1)), degrees=True)

    # Set Mode
    cur_mode = "hard_ctrl"

    # Initialize EnvExpert
    robot_init_qpos = get_robot_init_qpos(start_frame=start_frame, obj_type=obj_type)
    env_expert = EnvExpert(robot_init_qpos, start_frame, end_frame, obj_type=obj_type, render=False)
    env_expert.env.render_on(record=False, video_path="expert.mp4")

    state = env_expert.reset(mode=cur_mode, wait_steps=0)
    done, success = False, False
    env_expert.dummy_loop(render=False, debug=False)

    env_expert.env.render_off()

    sys.exit(0)

