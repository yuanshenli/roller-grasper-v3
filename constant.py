import numpy as np

"""
Robot Env
"""
MAX_ENV_STEPS = 600
SCALE_ERROR_POS = 200
SCALE_ERROR_ROT = 20

"""
Roller ID
"""
ROBOT_BASE_ID = 0
FRONT_ROLLER_ID = 4
LEFT_ROLLER_ID = 10
RIGHT_ROLLER_ID = 16
PALM_ROLLER_ID = 21
FLOOR_ID = 24
CUBE_ID = 25

ROLLER_ID_LIST = [FRONT_ROLLER_ID, LEFT_ROLLER_ID, RIGHT_ROLLER_ID, PALM_ROLLER_ID]

"""
Motor Limit
"""
MAX_REL_ROLLER = 0.2 * np.pi / 180.0
MAX_REL_PIVOT = 1.0 * np.pi / 180.0
MAX_REL_BASE = 0.08 * np.pi / 180.0
MAX_REL_PALM = 0.0002

MAX_REL_ACTION = np.array([MAX_REL_BASE, MAX_REL_PIVOT, MAX_REL_ROLLER,
                           MAX_REL_BASE, MAX_REL_PIVOT, MAX_REL_ROLLER,
                           MAX_REL_BASE, MAX_REL_PIVOT, MAX_REL_ROLLER,
                           MAX_REL_PALM, MAX_REL_PIVOT, MAX_REL_ROLLER])
MIN_REL_ACTION = -MAX_REL_ACTION

MAX_ABS_ROLLER = 6 * np.pi
MAX_ABS_PIVOT = np.pi
MAX_ABS_BASE = 15 * np.pi / 180.0   # 10
MAX_ABS_PALM = 0.02

MIN_ABS_ROLLER = -6 * np.pi
MIN_ABS_PIVOT = -np.pi
MIN_ABS_BASE = -10 * np.pi / 180.0  # -5
MIN_ABS_PALM = -0.02

MAX_ABS_ACTION = np.array([MAX_ABS_BASE, MAX_ABS_PIVOT, MAX_ABS_ROLLER,
                           MAX_ABS_BASE, MAX_ABS_PIVOT, MAX_ABS_ROLLER,
                           MAX_ABS_BASE, MAX_ABS_PIVOT, MAX_ABS_ROLLER,
                           MAX_ABS_PALM, MAX_ABS_PIVOT, MAX_ABS_ROLLER])

MIN_ABS_ACTION = np.array([MIN_ABS_BASE, MIN_ABS_PIVOT, MIN_ABS_ROLLER,
                           MIN_ABS_BASE, MIN_ABS_PIVOT, MIN_ABS_ROLLER,
                           MIN_ABS_BASE, MIN_ABS_PIVOT, MIN_ABS_ROLLER,
                           MIN_ABS_PALM, MIN_ABS_PIVOT, MIN_ABS_ROLLER])

CONTACT_CORRECTION_ACTION = -0.3 * np.pi / 180

"""
MuJoCo Render Parameters
"""
RUN_SPEED = 1
SLOW_RUN_SPEED = 0.05
HIDE_OVERLAY = False
DISPLAY_FRAME = True
RECORD_FPS = 20
CAM_AZIMUTH = 56
CAM_DISTANCE = 0.47
CAM_ELEVATION = -31

"""
MuJoCo Time Parameters
"""
UNIT_TIMESTEP = 0.0002

"""
MuJoCo Physics Parameters
"""
GRAVITY = np.array([0.0, 0.0, -9.8])
FRICTION = np.array([1.0, 0.001, 0.0001])

"""
MuJoCo Object Parameters
"""
OBJ_SIZE = np.array([0.034, 0.034, 0.034])
OBJ_TYPE = 'cube'
OBJ_SCALE = np.array([1, 1, 1])

"""
MuJoCo Mechanism Parameters
"""
# geom
ROBOT_POS = np.array([0.0, 0.0, 0.04])
FINGER_BASE_POS = np.array([[0.0, 0.048, 0.045],
                            [-0.04157, -0.034, 0.045],
                            [0.04157, -0.034, 0.045]])
FINGER_BASE_POS_OFFSET = np.array([[0.0, 0.02, 0.005],  # MuJoCo XML coordinate not located at real base axis!
                                   [-0.01732, -0.01, 0.005],
                                   [0.01732, -0.01, 0.005]])
FINGER_BASE_QUAT = np.array([[0.707, 0, 0, 0.707],
                             [-0.259, 0, 0, 0.966],
                             [-0.966, 0, 0, 0.259]])
FINGER_NORMAL = np.array([[0.0, -1.0, 0.0],
                          [0.8660, 0.5, 0.0],
                          [-0.8660, 0.5, 0.0]])
FINGER_LENGTH = 0.1235
PALM_BASE_POS = np.array([0.0, 0.0, 0.055])
PALM_LENGTH = 0.04
ROLLER_RADIUS = 0.0215

# gear ratio
FINGER_BASE_GEAR_RATIO = 3
PALM_BASE_GEAR_RATIO = 20
PIVOT_GEAR_RATIO = 10
ROLLER_GEAR_RATIO = 1
GEAR_RATIO = np.array([FINGER_BASE_GEAR_RATIO, PIVOT_GEAR_RATIO, ROLLER_GEAR_RATIO,
                       FINGER_BASE_GEAR_RATIO, PIVOT_GEAR_RATIO, ROLLER_GEAR_RATIO,
                       FINGER_BASE_GEAR_RATIO, PIVOT_GEAR_RATIO, ROLLER_GEAR_RATIO,
                       PALM_BASE_GEAR_RATIO, PIVOT_GEAR_RATIO, ROLLER_GEAR_RATIO])

# damping ratio
BASE_DAMPING = 0.1
PALM_BASE_DAMPING = 0.3
PIVOT_DAMPING = 0.1
ROLLER_DAMPING = 0.1
OBJ_DAMPING = 0

"""
MuJoCo Control Parameters
"""
USE_PD = True

BASE_KP = 0.75
PIVOT_KP = 1.0
ROLLER_KP = 1.0
PALM_KP = 0.75

BASE_KV = 3.0
PIVOT_KV = 0.0005
ROLLER_KV = 0.0005
PALM_KV = 10.0
