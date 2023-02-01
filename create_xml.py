from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import numpy as np
import sys
import glob
import os
from config import *
from constant import *
from utils import *
from mujoco_py import load_model_from_path


class XmlMaker:
    def __init__(self, init_pos=np.array([0.0, 0.0, 0.19]), init_quat=np.array([1, 0, 0, 0]), obj_type=OBJ_TYPE,
                 obj_fixed=False):
        self.finger_names = ["front", "left", "right", "palm"]

        # simulation
        self.unit_timestep = UNIT_TIMESTEP

        # physics
        self.gravity = GRAVITY
        self.friction = FRICTION

        # mechanism
        self.robot_pos = ROBOT_POS
        self.finger_base_pos = FINGER_BASE_POS + FINGER_BASE_POS_OFFSET
        self.finger_base_quat = FINGER_BASE_QUAT
        self.finger_normal = FINGER_NORMAL
        self.finger_length = FINGER_LENGTH
        self.palm_base_pos = PALM_BASE_POS
        self.palm_length = PALM_LENGTH
        self.roller_radius = ROLLER_RADIUS

        self.finger_base_gear_ratio = FINGER_BASE_GEAR_RATIO
        self.palm_base_gear_ratio = PALM_BASE_GEAR_RATIO
        self.pivot_gear_ratio = PIVOT_GEAR_RATIO
        self.roller_gear_ratio = ROLLER_GEAR_RATIO

        # dynamics
        self.base_damping = BASE_DAMPING
        self.palm_base_damping = PALM_BASE_DAMPING
        self.pivot_damping = PIVOT_DAMPING
        self.roller_damping = ROLLER_DAMPING
        self.obj_damping = OBJ_DAMPING

        # pd control
        self.use_pd = USE_PD

        self.base_kp = BASE_KP
        self.pivot_kp = PIVOT_KP
        self.roller_kp = ROLLER_KP
        self.palm_kp = PALM_KP

        self.base_kv = BASE_KV
        self.pivot_kv = PIVOT_KV
        self.roller_kv = ROLLER_KV
        self.palm_kv = PALM_KV

        # object
        self.obj_fixed = obj_fixed
        self.obj_pos = init_pos
        self.obj_quat = init_quat
        self.obj_size = OBJ_SIZE
        self.obj_type = obj_type
        self.obj_scale = OBJ_SCALE
        self.mesh_list = []

    def generate_xml(self, filename):
        mjc = Element("mujoco")  # root
        # option
        option = SubElement(mjc, "option")
        self.xml_set_attributes(option, [["timestep", str(self.unit_timestep)],
                                         ["cone", "elliptic"],
                                         ["collision", "all"],
                                         ["solver", "Newton"],
                                         ["gravity", np.array2string(self.gravity, separator=" ")[1:-1]]])
        # visual
        visual = SubElement(mjc, "visual")
        scale = SubElement(visual, "scale")
        self.xml_set_attributes(scale, [["framelength", "0.5"], ["framewidth", "0.01"]])

        # asset
        if self.obj_type in ["cube", "rectangular2", "rectangular"]:
            asset = SubElement(mjc, "asset")
            mesh = SubElement(asset, "mesh")
            self.xml_set_attributes(mesh, [["name", self.obj_type],
                                           ["file", "../meshes/" + self.obj_type + ".stl"],
                                           ["scale", np.array2string(self.obj_scale, separator=" ")[1:-1]]])
        elif self.obj_type in ["cube_w_opening", "mug", "prism_w_handle"]:
            # cnt = 0
            # while True:
            #     cnt += 1
            #     if os.path.exists("meshes/" + self.obj_type + "/" + self.obj_type + "_" + str(cnt) + ".stl"):
            #         asset = SubElement(mjc, "asset")
            #         mesh = SubElement(asset, "mesh")
            #         self.xml_set_attributes(mesh, [["name", self.obj_type + "_" + str(cnt)],
            #                                        ["file", "../meshes/" + self.obj_type + "/" +
            #                                         self.obj_type + "_" + str(cnt) + ".stl"],
            #                                        ["scale", np.array2string(self.obj_scale, separator=" ")[1:-1]]])
            #         self.mesh_list.append(self.obj_type + "_" + str(cnt))
            #     else:
            #         break
            for file in os.listdir(os.path.join("meshes", self.obj_type)):
                if file.endswith(".stl"):
                    asset = SubElement(mjc, "asset")
                    mesh = SubElement(asset, "mesh")
                    mesh_name = file.replace(".stl", "")
                    self.xml_set_attributes(mesh, [["name", mesh_name],
                                                   ["file", "../meshes/" + self.obj_type + "/" + file],
                                                   ["scale", np.array2string(self.obj_scale, separator=" ")[1:-1]]])
                    self.mesh_list.append(mesh_name)


        # worldbody
        wb = SubElement(mjc, "worldbody")
        self.xml_c1_worldbody(wb)

        # default
        df = SubElement(mjc, "default")
        geom = SubElement(df, "geom")
        self.xml_set_attributes(geom, [["contype", "1"],
                                       ["conaffinity", "1"],
                                       ["condim", "3"],
                                       # ["friction", np.array2string(self.friction, separator=" ")[1:-1]],
                                       ["solref", "0.001 1.5"],
                                       ["solimp", "0.95 0.95 .01"]
                                       ])

        # sensor
        sensor = SubElement(mjc, "sensor")
        self.xml_c1_sensor(sensor)

        # actuators
        actuator = SubElement(mjc, "actuator")
        self.xml_c1_actuator(actuator)

        # write xml file
        model = self.write_xml_file(mjc, filename)

        return mjc, model

    def xml_c1_worldbody(self, wb):
        robot = SubElement(wb, "body")
        self.xml_c2_robot(robot)
        # add fingers
        for ii, f_name in enumerate(self.finger_names[:-1]):
            pos = np.array2string(self.finger_base_pos[ii, :], separator=" ")[1:-1]
            quat = np.array2string(self.finger_base_quat[ii, :], separator=" ")[1:-1]
            self.xml_c2_finger(robot, f_name, pos, quat)

        # actuated palm
        self.xml_c2_palm(robot)
        floor = SubElement(wb, "body")
        self.xml_c2_floor(floor)
        cube = SubElement(wb, "body")
        self.xml_c2_cube(cube)

        # origin_points = np.array(np.meshgrid([0.034, -0.034], [0.034, -0.034], [0.034, -0.034], [1])).reshape(4,-1)  # 4x8
        # start_quat = self.obj_quat
        # rot_mat = R.from_quat(mjcquat_to_sciquat(start_quat)).as_matrix()
        # T_mat = np.eye(4)
        # T_mat[:3, :3] = rot_mat
        # T_mat[:3, 3] = self.obj_pos
        # # print("origin", origin_points)
        # # print(T_mat)
        # points = T_mat @ origin_points
        # # print(points)
        # new_points = points.T[:, :3]
        #
        # for i in range(len(new_points)):
        #     point = SubElement(wb, "body")
        #     self.xml_c2_marker(point, i, new_points[i])

    def xml_c1_sensor(self, sensor):
        for f_name in self.finger_names:  # finger 1 2 3 ..
            # joint sensors
            for ii in range(3):  # joint on each finger
                jointpos = SubElement(sensor, "jointpos")
                self.xml_set_attributes(jointpos, [["name", "j" + str(ii + 1) + "_" + f_name],
                                                   ["joint", "r" + str(ii + 1) + "_" + f_name]])

        # sphere pos
        for f_name in self.finger_names:  # finger 1 2 3 ..
            framepos = SubElement(sensor, "framepos")
            self.xml_set_attributes(framepos,
                                    [["name", "sphere_" + f_name], ["objtype", "body"], ["objname", "l3_" + f_name]])
        # box pos
        framepos = SubElement(sensor, "framepos")
        self.xml_set_attributes(framepos, [["name", "boxpos"], ["objtype", "body"], ["objname", "cube"]])
        framequat = SubElement(sensor, "framequat")
        self.xml_set_attributes(framequat, [["name", "boxorient"], ["objtype", "body"], ["objname", "cube"]])

    def xml_c1_actuator(self, actuator):
        # base joint
        for f_name in self.finger_names[:-1]:
            position = SubElement(actuator, "position")
            self.xml_set_attributes(position, [["name", "motor_base_" + f_name],
                                               ["kp", str(self.base_kp)],
                                               ["gear", str(self.finger_base_gear_ratio)],
                                               ["joint", "r1_" + f_name]])

            position = SubElement(actuator, "position")
            self.xml_set_attributes(position, [["name", "motor_pivot_" + f_name],
                                               ["kp", str(self.pivot_kp)],
                                               ["ctrllimited", "false"],
                                               ["gear", str(self.pivot_gear_ratio)],
                                               ["joint", "r2_" + f_name]])

            position = SubElement(actuator, "position")
            self.xml_set_attributes(position, [["name", "motor_roller_" + f_name],
                                               ["kp", str(self.roller_kp)],
                                               ["ctrllimited", "false"],
                                               ["gear", "1"],
                                               ["joint", "r3_" + f_name]])

        f_name = self.finger_names[-1]
        position = SubElement(actuator, "position")
        self.xml_set_attributes(position, [["name", "motor_base_" + f_name],
                                           ["kp", str(self.palm_kp)],
                                           ["gear", str(self.palm_base_gear_ratio)],
                                           ["joint", "r1_" + f_name]])
        position = SubElement(actuator, "position")
        self.xml_set_attributes(position, [["name", "motor_pivot_" + f_name],
                                           ["kp", str(self.pivot_kp)],
                                           ["ctrllimited", "false"],
                                           ["gear", str(self.pivot_gear_ratio)],
                                           ["joint", "r2_" + f_name]])
        position = SubElement(actuator, "position")
        self.xml_set_attributes(position, [["name", "motor_roller_" + f_name],
                                           ["kp", str(self.roller_kp)],
                                           ["ctrllimited", "false"],
                                           ["gear", "1"],
                                           ["joint", "r3_" + f_name]])

        # velocity control
        if self.use_pd:
            for f_name in self.finger_names[:-1]:
                velocity = SubElement(actuator, "velocity")

                self.xml_set_attributes(velocity, [["name", "motor_base_v_" + f_name],
                                                   ["kv", str(self.base_kv)],
                                                   ["gear", "1"],
                                                   ["joint", "r1_" + f_name]])

                velocity = SubElement(actuator, "velocity")

                self.xml_set_attributes(velocity, [["name", "motor_pivot_v_" + f_name],
                                                   ["kv", str(self.pivot_kv)],
                                                   ["ctrllimited", "false"],
                                                   ["gear", "1"],
                                                   ["joint", "r2_" + f_name]])

                velocity = SubElement(actuator, "velocity")

                self.xml_set_attributes(velocity, [["name", "motor_roller_v_" + f_name],
                                                   ["kv", str(self.roller_kv)],
                                                   ["ctrllimited", "false"],
                                                   ["gear", "1"],
                                                   ["joint", "r3_" + f_name]])

            f_name = self.finger_names[-1]
            velocity = SubElement(actuator, "velocity")
            self.xml_set_attributes(velocity, [["name", "motor_base_v_" + f_name],
                                               ["kv", str(self.palm_kv)],
                                               ["gear", "1"],
                                               ["joint", "r1_" + f_name]])

            velocity = SubElement(actuator, "velocity")
            self.xml_set_attributes(velocity, [["name", "motor_pivot_v_" + f_name],
                                               ["kv", str(self.pivot_kv)],
                                               ["ctrllimited", "false"],
                                               ["gear", "10"],
                                               ["joint", "r2_" + f_name]])

            velocity = SubElement(actuator, "velocity")
            self.xml_set_attributes(velocity, [["name", "motor_roller_v_" + f_name],
                                               ["kv", str(self.roller_kv)],
                                               ["ctrllimited", "false"],
                                               ["gear", "1"],
                                               ["joint", "r3_" + f_name]])

    def xml_c2_palm(self, robot):
        l1 = SubElement(robot, "body")
        pos = np.array2string(self.palm_base_pos, separator=" ")[1:-1]
        self.xml_set_attributes(l1, [["name", "l1_" + self.finger_names[-1]], ["pos", pos]])
        joint = SubElement(l1, "joint")
        self.xml_set_attributes(joint, [["axis", "0 0 1"],
                                        ["damping", str(self.palm_base_damping)],
                                        ["name", "r1_palm"],
                                        ["type", "slide"]])
        geom = SubElement(l1, "geom")
        self.xml_set_attributes(geom, [["name", "r1_palm"],
                                       ["rgba", "0 0 1 1"],
                                       ["size", "0.005 0.005"],
                                       ["type", "cylinder"]])

        l2 = SubElement(l1, "body")
        self.xml_set_attributes(l2, [["name", "l2_" + self.finger_names[-1]], ["pos", "-0.0 0.0 0.005"]])
        joint = SubElement(l2, "joint")
        self.xml_set_attributes(joint, [["axis", "0 0 1"],
                                        ["damping", str(self.pivot_damping)],
                                        ["name", "r2_" + self.finger_names[-1]],
                                        ["pos", "0 0 0"],
                                        ["type", "hinge"]])
        geom = SubElement(l2, "geom")
        self.xml_set_attributes(geom, [["mass", "0.01"],
                                       ["size", "0.025 0.005 0.005"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])

        l3 = SubElement(l2, "body")
        self.xml_set_attributes(l3, [["name", "l3_" + self.finger_names[-1]], ["pos", "0 0 0.035"]])
        joint = SubElement(l3, "joint")
        self.xml_set_attributes(joint, [["axis", "1 0 0"],
                                        ["damping", str(self.roller_damping)],
                                        ["name", "r3_" + self.finger_names[-1]],
                                        ["pos", "0 0 0"],
                                        ["type", "hinge"]])
        geom = SubElement(l3, "geom")
        self.xml_set_attributes(geom, [["name", "l3_" + self.finger_names[-1]],
                                       ["mass", "0.01"],
                                       ["size", str(self.roller_radius)],
                                       ["rgba", "1 0 1 1"],
                                       ["type", "sphere"]])
        l31 = SubElement(l2, "body")
        self.xml_set_attributes(l31, [["name", "l31_" + self.finger_names[-1]],
                                      ["pos",
                                       str(-self.roller_radius - 0.002) + " 0.0 " + str(self.roller_radius + 0.0015)]])
        geom = SubElement(l31, "geom")
        self.xml_set_attributes(geom, [["mass", "0.005"],
                                       ["size", "0.0015 0.002 0.018"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])
        l32 = SubElement(l2, "body")
        self.xml_set_attributes(l32, [["name", "l32_" + self.finger_names[-1]],
                                      ["pos",
                                       str(self.roller_radius + 0.002) + " 0.0 " + str(self.roller_radius + 0.0015)]])
        geom = SubElement(l32, "geom")
        self.xml_set_attributes(geom, [["mass", "0.005"],
                                       ["size", "0.0015 0.002 0.018"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])

    # robot base
    def xml_c2_robot(self, robot):
        robot.set("name", "robot")
        robot.set("pos", "0 0 0.04")

        geom = SubElement(robot, "geom")
        self.xml_set_attributes(geom, [["mass", "1.0"],
                                       ["pos", "0 0 0"],
                                       ["rgba", "1 0 0 0.1"],
                                       ["size", "0.2 0.02"],
                                       ["type", "cylinder"]])

        camera = SubElement(robot, "camera")

        camera_pos_x = 0
        camera_pos_y = 0
        camera_pos_z = 1
        camera_axis = np.array([1.0, -0.5, 0.0])
        camera_axis /= np.linalg.norm(camera_axis)  # normalize
        camera_theta = 90.0 * np.pi / 180.0
        camera_quat_w = str(np.cos(camera_theta / 2))
        camera_quat_x = str(camera_axis[0] * np.sin(camera_theta / 2))
        camera_quat_y = str(camera_axis[1] * np.sin(camera_theta / 2))
        camera_quat_z = str(camera_axis[2] * np.sin(camera_theta / 2))

        camera_pos = str(camera_pos_x) + " " + str(camera_pos_y) + " " + str(camera_pos_z)
        camera_quat = camera_quat_w + " " + camera_quat_x + " " + camera_quat_y + " " + camera_quat_z

        self.xml_set_attributes(camera, [["quat", camera_quat],
                                         ["fovy", "40"],
                                         ["name", "rgbd"],
                                         ["pos", camera_pos]])

    def xml_c2_finger(self, robot, f_name, pos, quat):
        l1 = SubElement(robot, "body")
        self.xml_set_attributes(l1, [["name", "l1_" + f_name], ["pos", pos], ["quat", quat]])

        joint = SubElement(l1, "joint")
        self.xml_set_attributes(joint, [["axis", "0 1 0"],
                                        ["damping", str(self.base_damping)],
                                        ["name", "r1_" + f_name],
                                        ["pos", "-0.02 0 -0.005"],
                                        ["type", "hinge"]])
        geom = SubElement(l1, "geom")
        self.xml_set_attributes(geom, [["mass", "0.01"],
                                       ["size", "0.02 0.015 0.005"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])

        offset = str((self.finger_length - 0.005) / 2)
        l11 = SubElement(l1, "body")
        self.xml_set_attributes(l11, [["name", "l11_" + f_name], ["pos", "0.025 0.0 " + offset]])
        geom = SubElement(l11, "geom")
        self.xml_set_attributes(geom, [["mass", "0.01"],
                                       ["size", "0.005 0.015 0.06325"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])

        l2 = SubElement(l11, "body")
        self.xml_set_attributes(l2, [["name", "l2_" + f_name], ["pos", "-0.01 0.0 " + offset]])
        joint = SubElement(l2, "joint")
        self.xml_set_attributes(joint, [["axis", "1 0 0"],
                                        ["damping", str(self.pivot_damping)],
                                        ["name", "r2_" + f_name],
                                        ["pos", "0 0 0"],
                                        ["type", "hinge"]])
        geom = SubElement(l2, "geom")
        self.xml_set_attributes(geom, [["mass", "0.01"],
                                       ["size", "0.005 0.005 0.025"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])

        l3 = SubElement(l2, "body")
        self.xml_set_attributes(l3, [["name", "l3_" + f_name], ["pos", "-0.035 0 0"]])
        joint = SubElement(l3, "joint")
        self.xml_set_attributes(joint, [["axis", "0 0 1"],
                                        ["damping", str(self.roller_damping)],
                                        ["name", "r3_" + f_name],
                                        ["pos", "0 0 0"],
                                        ["type", "hinge"]])
        geom = SubElement(l3, "geom")
        self.xml_set_attributes(geom, [["name", "l3_" + f_name],
                                       ["mass", "0.01"],
                                       ["size", str(self.roller_radius)],
                                       ["rgba", "1 0 1 1"],
                                       ["type", "sphere"]])
        l31 = SubElement(l2, "body")
        self.xml_set_attributes(l31, [["name", "l31_" + f_name],
                                      ["pos",
                                       str(-self.roller_radius - 0.0015) + " 0.0 " + str(-self.roller_radius - 0.002)]])
        geom = SubElement(l31, "geom")
        self.xml_set_attributes(geom, [["mass", "0.005"],
                                       ["size", "0.018 0.002 0.0015"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])
        l32 = SubElement(l2, "body")
        self.xml_set_attributes(l32, [["name", "l32_" + f_name],
                                      ["pos",
                                       str(-self.roller_radius - 0.0015) + " 0.0 " + str(self.roller_radius + 0.002)]])
        geom = SubElement(l32, "geom")
        self.xml_set_attributes(geom, [["mass", "0.005"],
                                       ["size", "0.018 0.002 0.0015"],
                                       ["rgba", "0 0 1 1"],
                                       ["type", "box"]])

    # Floor
    def xml_c2_floor(self, floor):
        self.xml_set_attributes(floor, [["name", "floor"], ["pos", "0 0 0"]])

        geom = SubElement(floor, "geom")
        self.xml_set_attributes(geom, [["size", "1.0 1.0 0.02"], ["rgba", "0 1 0 1"], ["type", "box"]])

    def xml_c2_cube(self, cube):
        pos = np.array2string(self.obj_pos, separator=" ")[1:-1]
        quat = np.array2string(self.obj_quat, separator=" ")[1:-1]
        size = np.array2string(self.obj_size, separator=" ")[1:-1]
        self.xml_set_attributes(cube, [["name", "cube"], ["pos", pos], ["quat", quat]])

        if not self.obj_fixed:
            joint = SubElement(cube, "joint")
            self.xml_set_attributes(joint, [["damping", str(self.obj_damping)], ["name", "cube_"], ["pos", "0 0 0"],
                                            ["type", "free"]])

        if self.obj_type in ["box", "sphere", "cylinder", "ellipsoid"]:
            geom = SubElement(cube, "geom")
            self.xml_set_attributes(geom, [["name", "cube_geom"],
                                           # ["density", "150"],
                                           ["mass", "0.08"],
                                           ["size", size],
                                           ["rgba", "1 1 1 0.5"],
                                           ["type", self.obj_type]])
        elif self.obj_type in ["cube", "rectangular2", "rectangular"]:
            geom = SubElement(cube, "geom")
            self.xml_set_attributes(geom, [["name", "cube_geom"],
                                           # ["density", "150"],
                                           ["mass", "0.08"],
                                           ["rgba", "1 1 1 0.5"],
                                           ["type", "mesh"],
                                           ["mesh", self.obj_type]])
        elif self.obj_type in ["cube_w_opening", "mug", "prism_w_handle"]:
            for mesh in self.mesh_list:
                geom = SubElement(cube, "geom")
                self.xml_set_attributes(geom, [["name", mesh],
                                               # ["mass", "0.04"],
                                               ["density", "500"],
                                               ["rgba", "1 1 1 0.5"],
                                               ["type", "mesh"],
                                               ["mesh", mesh]])

    def xml_c2_marker(self, cube, i, pos):

        pos = np.array2string(pos, separator=" ")[1:-1]
        quat = np.array2string(self.obj_quat, separator=" ")[1:-1]
        self.xml_set_attributes(cube, [["name", "point" + str(i)], ["pos", pos], ["quat", quat]])

        # if not self.obj_fixed:
        #     joint = SubElement(cube, "joint")
        #     self.xml_set_attributes(joint, [["damping", str(self.obj_damping)], ["name", "cube_"], ["pos", "0 0 0"],
        #                                     ["type", "free"]])

        geom = SubElement(cube, "geom")
        rgba = np.array2string(np.array([i & 0x1, (i >> 1) & 0x1, (i >> 2) & 0x1, 1]), separator=" ")[1:-1]
        self.xml_set_attributes(geom, [["name", "point_geom" + str(i)],
                                       # ["density", "150"],
                                       ["mass", "0.08"],
                                       ["size", "0.004"],
                                       ["rgba", rgba],
                                       ["type", "sphere"]])


    def xml_set_attributes(self, link, l_attributes):
        for a in l_attributes:
            link.set(a[0], a[1])

    def write_xml_file(self, my_data, file_name):
        xmlstr = minidom.parseString(tostring(my_data)).toprettyxml(indent="  ")
        with open(file_name, "w+") as f:
            # fcntl.flock(f, fcntl.LOCK_EX)
            f.write(xmlstr)
        model = load_model_from_path(file_name)
        return model
