import time
import sys
import math
import pickle
import copy
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread
import constants.unitree_legged_const as go2

'''
IMPORTANT: Turn off sport mode beforehand. Must suspend robot with harness.
Current functionality: sends hip joint positions in a sinusoidal wave and reads positions.
'''

PUB_FREQ = 250
PUB_PERIOD = 1/PUB_FREQ

crc = CRC()

joint_state_log = []

def read_joints(msg: LowState_):
    joint_state_log.append((time.time(), msg.motor_state))

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(read_joints, 10)
    
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0]=0xFE
    cmd.head[1]=0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q= go2.PosStopF
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].dq = go2.VelStopF
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

    # https://github.com/unitreerobotics/unitree_rl_gym/tree/main/resources/robots/go2/urdf
    # calf (2)
    # min_amp = -2.7227
    # max_amp = -0.83776
    # thigh (1)
    min_amp = -1.5708
    max_amp = 3.4907
    # hip (0)
    # min_amp = -1.0472
    # max_amp = 1.0472
    amplitude = (max_amp - min_amp) / 2
    offset = (max_amp + min_amp) / 2

    # adjusted to avoid large changes in angles
    if amplitude > 2.0944:
        freq = 1/8 * 2.0944/amplitude
    else:
        freq = 1/8

    # tuple with current time and joint commands
    joint_command_log = []

    # set initial position
    for name in ["RL_0", "RR_0", "FL_0", "FR_0"]:
        cmd.motor_cmd[go2.LegID[name]].mode = 0x01
        cmd.motor_cmd[go2.LegID[name]].q = 0
        cmd.motor_cmd[go2.LegID[name]].kp = 10.0 # Position (rad) control kp gain
        cmd.motor_cmd[go2.LegID[name]].dq = 0.0  # Target angular velocity(rad/s)
        cmd.motor_cmd[go2.LegID[name]].kd = 1.0  # Position (rad) control kd gain
        cmd.motor_cmd[go2.LegID[name]].tau = 0.0 # Feedforward toque 1N.m

    # motors_to_control = ["RL_2", "RR_2", "FL_2", "FR_2"]
    motors_to_control = ["RL_1"] #, "RR_1", "FL_1", "FR_1"]
    # motors_to_control = ["RL_0", "RR_0", "FL_0", "FR_0"]

    # set everything not q
    for name in motors_to_control:
        cmd.motor_cmd[go2.LegID[name]].mode = 0x01
        cmd.motor_cmd[go2.LegID[name]].kp = 10.0 # Position (rad) control kp gain
        cmd.motor_cmd[go2.LegID[name]].dq = 0.0  # Target angular velocity(rad/s)
        cmd.motor_cmd[go2.LegID[name]].kd = 1.0  # Position (rad) control kd gain
        cmd.motor_cmd[go2.LegID[name]].tau = 0.0 # Feedforward toque 1N.m

    time_0 = time.time()

    for i in range(10 * 300):
        start_time = time.time()
        sinusoidal_q = amplitude * math.sin(2 * math.pi * freq * (start_time - time_0)) + offset
        for name in motors_to_control:
            cmd.motor_cmd[go2.LegID[name]].q = sinusoidal_q  # Target angular(rad)
        
        cmd.crc = crc.Crc(cmd)

        #Publish message
        if pub.Write(cmd):
            joint_command_log.append((time.time(), [x.q for x in cmd.motor_cmd]))
        else:
            print("Waiting for subscriber.")
        end_time = time.time()
        if end_time - start_time < PUB_PERIOD:
            time.sleep(PUB_PERIOD - (end_time - start_time))

    pub.Close()
    sub.Close()

    print("Closed pub and sub.")

    log_file = "sinusoidal_logs.pkl"
    with open(log_file, 'wb') as f:
        pickle.dump(joint_command_log, f)
        pickle.dump(joint_state_log, f)
        pickle.dump({"amplitude": amplitude, "offset": offset, "freq": freq})

    print(f"Saved to {log_file}")
