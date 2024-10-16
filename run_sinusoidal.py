import time
import sys
import math

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread
import unitree_legged_const as go2

crc = CRC()

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    # https://support.unitree.com/home/en/developer/Basic_services
    
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0]=0xFE
    cmd.head[1]=0xEF
    # cmd.level_flag = 0xFF # reserved, not currently used
    cmd.gpio = 0 # ???

    frequency = 1/8
    amplitude = math.radians(30)
    # CF = 60 # Hz

    for i in range(20):
        # what does this do?
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode (working mode)
        cmd.motor_cmd[i].q= go2.PosStopF
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].dq = go2.VelStopF
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

    # Range of motion: https://github.com/LeCAR-Lab/dial-mpc/blob/140f0537de883e0f47abcfe26c560e5182fc3284/dial_mpc/models/unitree_go2/go2.xml#L16
    # https://github.com/unitreerobotics/unitree_rl_gym/blob/main/resources/robots/go2/urdf/go2.urdf

    # https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/low_level/lowlevel_control.py

    while True:

        current_time = time.time()
        
        # Calculate the sinusoidal joint positions
        sinusoidal_position = amplitude * math.sin(2 * math.pi * frequency * current_time)

        # Position(rad) control, set RL_0 rad front right hip, limits are -60 to 60
        cmd.motor_cmd[go2.LegID["RL_0"]].q = sinusoidal_position  # Target angular(rad)
        cmd.motor_cmd[go2.LegID["RL_0"]].kp = 1 # Position(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RL_0"]].dq = 0.0  # Target angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RL_0"]].kd = 0.5  # Position(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RL_0"]].tau = 0.0 # Feedforward toque 1N.m
        
        cmd.crc = crc.Crc(cmd)

        #Publish message
        if pub.Write(cmd):
            print("Publish success. msg:", cmd.crc)
        else:
            print("Waitting for subscriber.")

        time.sleep(0.002)