import time
import sys
import numpy as np
import pickle
import torch
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
import constants.unitree_legged_const as go2
import utils.client_utils as client_utils
import utils.publisher_utils as pub_utils
import utils.isaacgym_utils as isaac_utils
import examples.actor_critic as actor_critic

from scipy.spatial.transform import Rotation as R

PUB_FREQ = 50 # according to script
PUB_PERIOD = 1/PUB_FREQ
SCALE_FACTOR = 0.25

SIT_POS = [-0.1, 1.1, -2.0, -0.1, 1.1, -2.0, -0.1, 1.1, -2.6, -0.1, 1.1, -2.6]

KP = 20
KD = 0.5

crc = CRC()

joint_state_log = []
pub_log = []
stop_program = False
sit = False
wrote_to_log_file = False

PREV_TARGET_Q = None
PREV_TARGET_Q_TIME = None

observations = {
    "base_ang_vel": None,
    "projected_gravity": None,
    "cmd_vel": None,
    "joint_pos_vel": None,
    "init_raw_action": None,
    "raw_actions": None
}

ORDER = ["FL_0", "FL_1", "FL_2", "FR_0", "FR_1", "FR_2", "RL_0", "RL_1", "RL_2", "RR_0", "RR_1", "RR_2"]

motor_qs_defaults = {
    "FR_0": 0.1,  # Front right hip
    "FR_1": 0.8,  # Front right thigh
    "FR_2": -1.5,  # Front right calf
    "FL_0": 0.1,
    "FL_1": 0.8,
    "FL_2": -1.5,
    "RR_0": 0.1,
    "RR_1": 1,
    "RR_2": -1.5,
    "RL_0": 0.1,
    "RL_1": 1,
    "RL_2": -1.5,
}

motor_qs_defaults_ordered = [motor_qs_defaults[order_name] for order_name in ORDER]

key_state = [
    ["R1", 0],
    ["L1", 0],
    ["start", 0],
    ["select", 0],
    ["R2", 0],
    ["L2", 0],
    ["F1", 0],
    ["F2", 0],
    ["A", 0],
    ["B", 0],
    ["X", 0],
    ["Y", 0],
    ["up", 0],
    ["right", 0],
    ["down", 0],
    ["left", 0],
]

def projected_gravity_vector(imu_quaternion):
    # Use rotation from quaternion to find proj g
    rotation = R.from_quat(imu_quaternion)
    gravity_vec_w = np.array([0.0, 0.0, -1.0])  # Gravity vector in world
    gravity_proj = -1 * rotation.apply(gravity_vec_w)
    return gravity_proj

def get_joint_pos_vel(msg):
    # motor_qs = [msg.motor_state[i].q for i in range(12)]
    motor_qs = [
        msg.motor_state[3].q,   # 0  -> FL_hip_joint   to FL_hip   -> 3
        msg.motor_state[4].q,   # 1  -> FR_hip_joint   to FR_hip   -> 0
        msg.motor_state[5].q,   # 2  -> RL_hip_joint   to RL_hip   -> 9
        msg.motor_state[0].q,   # 3  -> RR_hip_joint   to RR_hip   -> 6
        msg.motor_state[1].q,   # 4  -> FL_thigh_joint to FL_thigh -> 4
        msg.motor_state[2].q,   # 5  -> FR_thigh_joint to FR_thigh -> 1
        msg.motor_state[9].q,  # 6  -> RL_thigh_joint to RL_thigh -> 10
        msg.motor_state[10].q,   # 7  -> RR_thigh_joint to RR_thigh -> 7
        msg.motor_state[11].q,   # 8  -> FL_calf_joint  to FL_calf  -> 5
        msg.motor_state[6].q,   # 9  -> FR_calf_joint  to FR_calf  -> 2
        msg.motor_state[7].q,  # 10 -> RL_calf_joint  to RL_calf  -> 11
        msg.motor_state[8].q,   # 11 -> RR_calf_joint  to RR_calf  -> 8
    ]

    motor_dqs = [msg.motor_state[i].dq for i in range(12)]

    motor_dqs = [
        msg.motor_state[3].dq,   # 0  -> FL_hip_joint   to FL_hip   -> 3
        msg.motor_state[4].dq,   # 1  -> FR_hip_joint   to FR_hip   -> 0
        msg.motor_state[5].dq,   # 2  -> RL_hip_joint   to RL_hip   -> 9
        msg.motor_state[0].dq,   # 3  -> RR_hip_joint   to RR_hip   -> 6
        msg.motor_state[1].dq,   # 4  -> FL_thigh_joint to FL_thigh -> 4
        msg.motor_state[2].dq,   # 5  -> FR_thigh_joint to FR_thigh -> 1
        msg.motor_state[9].dq,  # 6  -> RL_thigh_joint to RL_thigh -> 10
        msg.motor_state[10].dq,   # 7  -> RR_thigh_joint to RR_thigh -> 7
        msg.motor_state[11].dq,   # 8  -> FL_calf_joint  to FL_calf  -> 5
        msg.motor_state[6].dq,   # 9  -> FR_calf_joint  to FR_calf  -> 2
        msg.motor_state[7].dq,  # 10 -> RL_calf_joint  to RL_calf  -> 11
        msg.motor_state[8].dq,   # 11 -> RR_calf_joint  to RR_calf  -> 8
    ]

    joint_pos = [q - default for q,
                     default in zip(motor_qs, motor_qs_defaults_ordered)]
    
    return joint_pos + motor_dqs

def read_joints(msg: LowState_):
    # thread-safety https://stackoverflow.com/a/18568017
    joint_state_log.append((time.time(), msg.motor_state))
    observations["base_ang_vel"] = np.array([
        msg.imu_state.gyroscope[0],
        msg.imu_state.gyroscope[1],
        msg.imu_state.gyroscope[2]
    ])

    imu_quaternion = np.array([
        msg.imu_state.quaternion[0],
        msg.imu_state.quaternion[1],
        msg.imu_state.quaternion[2],
        msg.imu_state.quaternion[3]
    ]) # Format: (0-w, 1-x, 2-y, 3-z)
    observations["projected_gravity"] = projected_gravity_vector(imu_quaternion)

    observations["cmd_vel"] = np.array([0,0,0])

    observations["joint_pos_vel"] = get_joint_pos_vel(msg)
    observations["init_raw_action"] = (np.array(observations["joint_pos_vel"][:12]) - np.array(motor_qs_defaults_ordered)) / SCALE_FACTOR

def WirelessControllerHandler(msg: WirelessController_):
    global stop_program
    global sit

    for i in range(16):
        key_state[i][1] = (msg.keys & (1 << i)) >> i

    if key_state[9][1] == 1: # key B
        print(f"Stop program!")
        stop_program = True
    elif key_state[8][1] == 1:
        print(f"Sit!")
        sit = True

def get_target_q(model):
    obs = np.concatenate(
        (
            [0,0,0],
            observations["base_ang_vel"],
            observations["projected_gravity"],
            observations["cmd_vel"],
            observations["joint_pos_vel"]
        )
    )

    if observations["raw_actions"] is not None:
        obs = np.concatenate((obs, observations["raw_actions"]))
    else:
        obs = np.concatenate((obs, observations["init_raw_action"]))

    obs = obs.astype(np.float32).reshape(1, -1)

    # Run inference
    try:
        ort_outs = model.actor(torch.from_numpy(obs))
        # print(obs)
        print(ort_outs)
        observations["raw_actions"] = ort_outs[0].detach().numpy()
    except Exception as e:
        print(f"Inference failed: {e}")
        observations["raw_actions"] = np.zeros_like(observations["init_raw_action"])

    # Accounting for offset and scale (Isaac Lab)
    processed_actions = (observations["raw_actions"] * SCALE_FACTOR + motor_qs_defaults_ordered).tolist()
    # processed_actions_ordered = processed_actions
    processed_actions_ordered = [
        processed_actions[3],   # 0  -> FL_hip_joint   to FL_hip   -> 3
        processed_actions[4],   # 1  -> FR_hip_joint   to FR_hip   -> 0
        processed_actions[5],   # 2  -> RL_hip_joint   to RL_hip   -> 9
        processed_actions[0],   # 3  -> RR_hip_joint   to RR_hip   -> 6
        processed_actions[1],   # 4  -> FL_thigh_joint to FL_thigh -> 4
        processed_actions[2],   # 5  -> FR_thigh_joint to FR_thigh -> 1
        processed_actions[9],  # 6  -> RL_thigh_joint to RL_thigh -> 10
        processed_actions[10],   # 7  -> RR_thigh_joint to RR_thigh -> 7
        processed_actions[11],   # 8  -> FL_calf_joint  to FL_calf  -> 5
        processed_actions[6],   # 9  -> FR_calf_joint  to FR_calf  -> 2
        processed_actions[7],  # 10 -> RL_calf_joint  to RL_calf  -> 11
        processed_actions[8],   # 11 -> RR_calf_joint  to RR_calf  -> 8
    ]

    print(f"{processed_actions}")

    # # Motor limits
    # motor_limits_ordered = [
    #     [-0.837, 0.837],  # Front Hip
    #     [-3.490, 1.570],  # Front Thigh
    #     [-2.720, 0.837],  # Front Calf
    #     [-0.837, 0.837],  # Front Hip
    #     [-3.490, 1.570],  # Front Thigh
    #     [-2.720, 0.837],  # Front Calf
    #     [-0.837, 0.837],  # Rear Hip
    #     [-4.530, 1.570],  # Rear Thigh
    #     [-2.720, 0.837],  # Rear Calf
    #     [-0.837, 0.837],  # Rear Hip
    #     [-4.530, 1.570],  # Rear Thigh
    #     [-2.720, 0.837]   # Rear Calf
    # ]

    # Clipping actions by a scale of the motor limits:
    clipped_actions_ordered = [0]*12
    for i, action in enumerate(processed_actions_ordered):
        min_limit, max_limit = list(go2.JOINT_LIMITS.values())[i]

        # Applying scale factor
        min_limit = min_limit * 0.95
        max_limit = max_limit * 0.95
        clipped_actions_ordered[i] = max(min(action, max_limit), min_limit)

        # also limit change in joint angle (limit to let's say 10 rad/s?)
        # graph showed by Lukas shows fastest slope at 24 rad/s
        print(f"ANGLES BEFORE CLIPPING: {clipped_actions_ordered}")
        if PREV_TARGET_Q is not None and PREV_TARGET_Q_TIME is not None:
            max_angle_change = 10 * (time.time() - PREV_TARGET_Q_TIME)
            clipped_actions_ordered[i] = min(max(clipped_actions_ordered[i], PREV_TARGET_Q[i] -max_angle_change), PREV_TARGET_Q[i] +max_angle_change)
        print(f"ANGLES AFTER CLIPPING: {clipped_actions_ordered}")

    # https://github.com/eppl-erau-db/go2_rl_ws/blob/146a64d9cec414ead91775fe2d43c722edc7c649/src/rl_deploy/src/go2_rl_control_cpp.cpp#L256
    return clipped_actions_ordered

if __name__ == '__main__':

    model_path = "models/model_1500.pt"

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]

    model = actor_critic.ActorCritic(
        num_actor_obs=48,
        num_critic_obs=48,
        num_actions=12,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation='elu',
        init_noise_std=init_noise_std
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

    model.eval()

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    rsc = client_utils.get_robot_state_client()
    client_utils.set_service(rsc, "sport_mode", False)

    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(read_joints, 10)

    sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    sub.Init(WirelessControllerHandler, 10)
    
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0]=0xFE
    cmd.head[1]=0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = go2.PosStopF
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].dq = go2.VelStopF
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

    start_time = time.perf_counter()
    period_index = 0

    while len(joint_state_log) == 0:
        print(f"Waiting for subscriber to get a data point.")

    while True:
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        if stop_program or sit:
            # https://github.com/eppl-erau-db/go2_rl_ws/blob/146a64d9cec414ead91775fe2d43c722edc7c649/src/rl_deploy/src/go2_rl_control_cpp.cpp#L280
            for motor, id in go2.LegID.items():
                if stop_program:
                    cmd.motor_cmd[id].q = joint_state_log[-1][1][id].q
                    cmd.motor_cmd[id].dq = 0.0
                    cmd.motor_cmd[id].kp = 40
                    cmd.motor_cmd[id].kd = 5
                    cmd.motor_cmd[id].tau = 0.0
                elif sit:
                    cmd.motor_cmd[id].q = SIT_POS[id]
                    cmd.motor_cmd[id].dq = 0.0
                    cmd.motor_cmd[id].kp = 30
                    cmd.motor_cmd[id].kd = 10
                    cmd.motor_cmd[id].tau = 0.0

            cmd.crc = crc.Crc(cmd)
            # if not pub.Write(cmd):
            #     print(f"Unable to publish cmd to damping!")
            # else:
            #     pub_log.append((time.time(), cmd.motor_cmd))

            if not wrote_to_log_file:
                log_file = "model_logs.pkl"
                with open(log_file, 'wb') as f:
                    pickle.dump(pub_log, f)
                    pickle.dump(joint_state_log, f)
                print(f"Saved to {log_file} after stopping program.")
                wrote_to_log_file = True

            continue

        if elapsed_time >= PUB_PERIOD * (period_index + 1):
            # run model
            target_q = get_target_q(model)
            PREV_TARGET_Q = target_q
            PREV_TARGET_Q_TIME = time.time()
            for motor, id in go2.LegID.items():
                cmd.motor_cmd[id].q = target_q[id]
                cmd.motor_cmd[id].dq = 0.0
                cmd.motor_cmd[id].kp = KP
                cmd.motor_cmd[id].kd = KD
                cmd.motor_cmd[id].tau = 0.0

            cmd.crc = crc.Crc(cmd)

            if not pub.Write(cmd):
                print(f"Unable to publish cmd!")
            else:
                pub_log.append((time.time(), cmd.motor_cmd))

            period_index += 1
        else:
            print(f"Looped too fast, wait.")

        time_to_sleep = PUB_PERIOD * (period_index + 1) - (time.perf_counter() - start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    pub.close()
    sub.close()
