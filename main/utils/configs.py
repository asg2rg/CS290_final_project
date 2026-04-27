EVAL = False

# gym setup
WINDOW_H = 500
WINDOW_W = 1200
CAM_X = 250
RENDER = False

    # road
ROAD_CNT = 4
ROAD_W = 80
ROAD_L = 5000
ROAD_TOP = 90
    # car
CAR_L = 40
CAR_W = 24

# env setup
DISCRETE = False
TARGET_SPEED = 100.0
MIN_SPEED = 0.0
MAX_SPEED = 100.0
TURN_UNIT = 1.0
CLAMP = False

OBS_DIM = 8
ACTION_DIM = 2
STACK_SZ = 4

TIMESTEP = 0.2
CAR_INITIAL_VEL = 50.0
AGENT_1_INITIAL_VEL = 25.0

# behavior
MAX_ANG = 3.0
MAX_ACC = 10.0

TURN_SHAPE = 7
ACC_SHAPE = 7  # only takes effect in DISCRETE
turn_cmds = {
            0: -1.0, # slight left
            1: -2.0, # med left
            2: -3.0, # hard left
            3: 0.0,  # no turn
            4: 1.0,  # slight right
            5: 2.0,  # med right
            6: 3.0   # hard right
        }
acc_cmds = {
            0: 2.0,   # slight accel
            1: 5.0,   # med accel
            2: 10.0,  # hard accel
            3: 0.0,   # no accel
            4: -2.0,  # slight brake
            5: -5.0,  # med brake
            6: -10.0   # hard brake
        }

# model
EXPLORE_NOISE_STD = 0.3
EXPLORE_NOISE_MIN = 0.05

G_STEPS = 1_000_000
DECAY_INTERVAL = 1000

EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.9957

DISCOUNT = 0.99
BATCH_SIZE = 1024

def verify_setup(dec = EPS_DECAY):
    import math
    # check how many steps to decay from EPS_START to EPS_MIN with EPS_DECAY
    ratio = EPS_MIN / EPS_START
    steps = math.log(ratio, dec)
    final_explore_step = steps * DECAY_INTERVAL
    print(f"With decay {dec}, it would take {steps:.2f} decay steps, or {final_explore_step:.0f} global steps to reach EPS_MIN.")
