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

PUB_FREQ = 500

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

    freq = 1/8
    # min_amp = -2.7227
    # max_amp = -0.83776
    # thigh
    min_amp = -1.5708
    max_amp = 3.4907
    amplitude = (max_amp - min_amp) / 2
    offset = (max_amp + min_amp) / 2

    # tuple with current time and joint commands
    joint_command_log = []

    for i in range(10 * 300):
        current_time = time.time()
        sinusoidal_q = amplitude * math.sin(2 * math.pi * freq * current_time) + offset
        # # Poinstion(rad) control, set RL_0 rad
        for name in ["RL_0", "RR_0", "FL_0", "FR_0"]:
            cmd.motor_cmd[go2.LegID[name]].mode = 0x01
            cmd.motor_cmd[go2.LegID[name]].q = 0
            cmd.motor_cmd[go2.LegID[name]].kp = 10.0 # Poinstion(rad) control kp gain
            cmd.motor_cmd[go2.LegID[name]].dq = 0.0  # Target angular velocity(rad/ss)
            cmd.motor_cmd[go2.LegID[name]].kd = 1.0  # Poinstion(rad) control kd gain
            cmd.motor_cmd[go2.LegID[name]].tau = 0.0 # Feedforward toque 1N.m
        for name in ["RL_1", "RR_1", "FL_1", "FR_1"]:
        # for name in ["RL_2", "RR_2", "FL_2", "FR_2"]:
            cmd.motor_cmd[go2.LegID[name]].mode = 0x01
            if True: # name == "RL_0":
                cmd.motor_cmd[go2.LegID[name]].q = sinusoidal_q  # Target angular(rad)
            else:
                cmd.motor_cmd[go2.LegID[name]].q = 0  # Target angular(rad)
            cmd.motor_cmd[go2.LegID[name]].kp = 10.0 # Poinstion(rad) control kp gain
            cmd.motor_cmd[go2.LegID[name]].dq = 0.0  # Target angular velocity(rad/ss)
            cmd.motor_cmd[go2.LegID[name]].kd = 1.0  # Poinstion(rad) control kd gain
            cmd.motor_cmd[go2.LegID[name]].tau = 0.0 # Feedforward toque 1N.m
        
        cmd.crc = crc.Crc(cmd)

        #Publish message
        if pub.Write(cmd):
            joint_command_log.append((time.time(), [x.q for x in cmd.motor_cmd]))
        else:
            print("Waiting for subscriber.")

        time.sleep(1/PUB_FREQ)

    pub.Close()
    sub.Close()

    print("Closed pub and sub.")

    log_file = "sinusoidal_logs.pkl"
    with open(log_file, 'wb') as f:
        pickle.dump(joint_command_log, f)
        pickle.dump(joint_state_log, f)

    print(f"Saved to {log_file}")
