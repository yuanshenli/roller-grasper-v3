from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
from multiprocessing import Process
import numpy as np
import os
import glfw
from heuristic import *

from create_xml import XmlMaker
from hand import Finger, Palm
from config import *
from constant import *
from utils import *


class RobotEnv:
    def __init__(self, init_qpos, init_ctrl=None, obj_fixed=False,
                 render=False, record=False, obj_type=OBJ_TYPE, model_xml=None):
        self.obj_type = obj_type
        self.obj_fixed = obj_fixed
        if self.obj_type in ["cube", "rectangular2", "rectangular"]:
            self.obj_geom_cnt = 1
        else:
            self.obj_geom_cnt = self.get_obj_geom_cnt_from_type()
        # print("obj_geom_cnt", self.obj_geom_cnt)
        self.obj_geom_ids = np.arange(CUBE_ID, CUBE_ID + self.obj_geom_cnt)

        # xml
        if model_xml is None:
            if os.path.exists('xml') is False:
                os.mkdir('xml')
            self.model_xml = 'xml/rgv3.xml'
        else:
            self.model_xml = model_xml
        xml_maker = XmlMaker(init_pos=init_qpos[-7:-4], init_quat=init_qpos[-4:], obj_type=obj_type, obj_fixed=obj_fixed)
        # xml_maker.generate_xml(self.model_xml)
        # self.model = load_model_from_path(self.model_xml)
        _, self.model = xml_maker.generate_xml(self.model_xml)

        # sim
        self.sim = MjSim(self.model)
        self.record = record
        if render:
            self.create_viewer()
            self.render_on(record=record)
        else:
            self.viewer = None

        # finger and palm
        self.hand = [Finger('front'), Finger('left'), Finger('right'), Palm()]

        # env step
        self.timestep = 0

        # init
        self.robot_init_qpos = init_qpos[0:12]
        self.init_pos = init_qpos[-7:-4]
        self.init_quat = init_qpos[-4:]
        if init_ctrl is not None:
            self.init_ctrl = init_ctrl[0:12]
        else:
            self.init_ctrl = self.robot_init_qpos * GEAR_RATIO

        # reset
        for i in range(12):
            self.sim.data.qpos[i] = self.robot_init_qpos[i]
            self.sim.data.ctrl[i] = self.init_ctrl[i]
        # print(self.sim.data.qpos)
        # print(self.sim.data.ctrl)

        if not obj_fixed:
            for i in range(3):
                self.sim.data.qpos[i + 12] = self.init_pos[i]

            for i in range(4):
                self.sim.data.qpos[i + 15] = self.init_quat[i]

    def create_viewer(self, run_speed=RUN_SPEED):
        self.viewer = MjViewer(self.sim)
        self.viewer._run_speed = run_speed
        self.viewer._hide_overlay = HIDE_OVERLAY
        self.viewer.vopt.frame = DISPLAY_FRAME
        self.viewer.cam.azimuth = CAM_AZIMUTH
        self.viewer.cam.distance = CAM_DISTANCE
        self.viewer.cam.elevation = CAM_ELEVATION

    def is_drop(self, log_permit=False):
        if not self.obj_fixed:
            obj_pos = self.sim.data.qpos[-7:-4].copy()
            # print(obj_pos)
            if (obj_pos > 0.5).any():
                return True
        for j in range(len(self.sim.data.contact)):
            con = self.sim.data.contact[j]
            if con.geom1 in [ROBOT_BASE_ID, FLOOR_ID] and con.geom2 in self.obj_geom_ids:
                if log_permit:
                    log("The object is dropped!", "r", "B")
                return True  # The object is dropped
        return False

    def get_roller_contact(self):
        contact = np.zeros(4)
        ncon = self.sim.data.ncon
        for i in range(ncon):
            con = self.sim.data.contact[i]
            # print(con.geom1, con.geom2)
            if con.geom1 in ROLLER_ID_LIST and con.geom2 in self.obj_geom_ids:
                contact[ROLLER_ID_LIST.index(con.geom1)] = 1
        # print(contact)
        return contact

    def get_roller_contact_points(self):
        if self.obj_fixed:
            frame = np.hstack((self.init_pos, self.init_quat))
        else:
            frame = self.sim.data.qpos[-7:].copy()
        # print(frame)
        T_mat = np.eye(4)
        T_mat[:3, :3] = R.from_quat(mjcquat_to_sciquat(frame[-4:])).as_matrix()
        T_mat[:3, 3] = frame[:3]
        # print("T_mat", T_mat)
        # input()
        contact_points = np.full((4, 3), np.nan)    # N x 3
        ncon = self.sim.data.ncon
        for i in range(ncon):
            con = self.sim.data.contact[i]
            # print(con.geom1, con.geom2)
            # print(con.pos)
            if con.geom1 in ROLLER_ID_LIST and con.geom2 in self.obj_geom_ids:
                contact_points[ROLLER_ID_LIST.index(con.geom1)] = con.pos
        contact_points = np.hstack((contact_points, np.ones((4, 1)))).T
        contact_points = np.linalg.inv(T_mat) @ contact_points
        contact_points = contact_points[:3, :].T
        return contact_points

    def get_observation(self, contact=True):
        obs = self.sim.data.qpos.copy()
        if self.obj_fixed:
            obs = np.hstack((obs, self.init_pos, self.init_quat))
        if contact:
            con = self.get_roller_contact()
            obs = np.insert(obs, 12, con)
        return obs

    def render_on(self, record=False, video_path="./video.mp4", run_speed=RUN_SPEED):
        if self.viewer is None:
            self.create_viewer(run_speed=run_speed)
        if record:
            self.record = True
            self.viewer._record_video = True
            self.viewer._video_process = Process(target=save_video,
                                                 args=(self.viewer._video_queue, video_path, RECORD_FPS))
            self.viewer._video_process.start()

    def render_off(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            if self.record:
                self.record = False
                self.viewer._record_video = False
                self.viewer._video_queue.put(None)
                self.viewer._video_process.join()
        self.viewer = None

    def reset(self, init_qpos=None, init_ctrl=None, render=False, record=False):
        self.sim.reset()
        self.timestep = 0

        if init_qpos is not None:
            self.robot_init_qpos = init_qpos[0:12]
            if not self.obj_fixed:
                self.init_pos = init_qpos[-7:-4]
                self.init_quat = init_qpos[-4:]

        if init_ctrl is not None:
            self.init_ctrl = init_ctrl[0:12]
        else:
            self.init_ctrl = self.robot_init_qpos * GEAR_RATIO

        for i in range(12):
            self.sim.data.qpos[i] = self.robot_init_qpos[i]
            self.sim.data.ctrl[i] = self.init_ctrl[i]

        if not self.obj_fixed:
            for i in range(3):
                self.sim.data.qpos[i + 12] = self.init_pos[i]

            for i in range(4):
                self.sim.data.qpos[i + 15] = self.init_quat[i]

    def step(self, action, sample_interval=50, relative=np.tile(np.array([True, True, True]), 4), overlay=None):
        # action
        for i in range(len(action)):
            if relative[i]:
                self.sim.data.ctrl[i] += action[i]
            else:
                self.sim.data.ctrl[i] = action[i]
            self.sim.data.ctrl[i] = self.sim.data.ctrl[i].clip(MIN_ABS_ACTION[i] * GEAR_RATIO[i],
                                                               MAX_ABS_ACTION[i] * GEAR_RATIO[i])
        # print(self.sim.data.ctrl[0:12])

        # env step
        self.timestep += 1
        for i in range(sample_interval):
            self.sim.step()
            if self.viewer is not None:
                if overlay is not None:
                    self.viewer.add_overlay(const.GRID_TOPRIGHT, " ", overlay)
                self.viewer.render()

        return self.get_observation()

    def get_obj_geom_cnt_from_type(self):
        cnt = 0
        for file in os.listdir(os.path.join("meshes", self.obj_type)):
            if file.endswith(".stl"):
                cnt += 1
        return cnt


if __name__ == "__main__":
    start_waypoint = ((0.0, 0.0, 0.19), 0, (0, 0, 1))
    start_frame = calc_single_frame(start_waypoint)
    robot_init_qpos = calc_robot_init_qpos(obj_frame=start_frame, obj_r=0.03)
    env = RobotEnv(init_qpos=np.hstack((robot_init_qpos, start_frame)), render=True, record=False, obj_type='sphere')
    episode = 0
    while True:
        action = np.zeros(12)
        env.step(action)
        if env.timestep > MAX_ENV_STEPS:
            episode += 1
            print("episode: {}".format(episode))
            # env.render_off()
            # env.render_on(record=True, run_speed=5)
            env.reset()
