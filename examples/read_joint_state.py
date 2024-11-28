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
import utils.client_utils as client_utils

'''
IMPORTANT: Turn off sport mode beforehand. Must suspend robot with harness.
Current functionality: sends hip joint positions in a sinusoidal wave and reads positions.
'''

crc = CRC()

joint_state_log = []

def read_joints(msg: LowState_):
    joint_state_log.append((time.time(), msg))

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    rsc = client_utils.get_robot_state_client()
    client_utils.set_service(rsc, "sport_mode", False)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(read_joints, 10)

    import time

    t_end = time.time() + 60 * 1
    while time.time() < t_end:
        pass

    sub.Close()

    print("Closed pub and sub.")

    log_file = "joint_states.pkl"
    with open(log_file, 'wb') as f:
        print(joint_state_log[-1].motor_state)
        pickle.dump(joint_state_log, f)

    print(f"Saved to {log_file}")
