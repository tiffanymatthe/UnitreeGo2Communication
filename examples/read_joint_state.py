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

PUB_FREQ = 250
PUB_PERIOD = 1/250
DEFAULT_RAMP_RATE = 2 * math.pi / 180

SIT_POS = [-0.1, 1.1, -2.0, -0.1, 1.1, -2.0, -0.1, 1.1, -2.6, -0.1, 1.1, -2.6]

crc = CRC()

joint_state_log = []
joint_command_log = []

def move_to_initial_pose(cmd, pub, motors, target_positions):
    pause_counts = 500
    while len(joint_state_log) == 0:
        print(f"Waiting for subscriber to get a data point.")

    period_count = 1

    while True:
        current_time = time.perf_counter()
        for name in motors:
            latest_joint_states = joint_state_log[-1][1].motor_state
            if target_positions[name] < latest_joint_states[go2.LegID[name]].q:
                intermediate_target = max(latest_joint_states[go2.LegID[name]].q - DEFAULT_RAMP_RATE * PUB_PERIOD * period_count, target_positions[name])
            else:
                intermediate_target = min(latest_joint_states[go2.LegID[name]].q + DEFAULT_RAMP_RATE * PUB_PERIOD * period_count, target_positions[name])
            cmd.motor_cmd[go2.LegID[name]].mode = 0x01
            cmd.motor_cmd[go2.LegID[name]].q = intermediate_target
            cmd.motor_cmd[go2.LegID[name]].kp = 20.0
            cmd.motor_cmd[go2.LegID[name]].dq = 0.0
            cmd.motor_cmd[go2.LegID[name]].kd = 0.5
            cmd.motor_cmd[go2.LegID[name]].tau = 0.0

        cmd.crc = crc.Crc(cmd)
        if pub.Write(cmd):
            joint_command_log.append((time.time(), [x.q for x in cmd.motor_cmd]))
        else:
            print("Waiting for subscriber.")

        sleep_time = time.perf_counter() - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        period_count += 1

        latest_joint_states = joint_state_log[-1][1].motor_state
        reached_target = True
        for motor in motors:
            if not math.isclose(latest_joint_states[go2.LegID[motor]].q, target_positions[motor], abs_tol=0.1):
                # print(f"Want {latest_joint_states[go2.LegID[motor]].q} to match {target_positions[motor]}")
                reached_target = False
        if reached_target:
            if pause_counts <= 0:
                cmd.crc = crc.Crc(cmd)
                if pub.Write(cmd):
                    joint_command_log.append((time.time(), [x.q for x in cmd.motor_cmd]))
                else:
                    print("Waiting for subscriber.")
                break
            else:
                pause_counts -= 1
    
    for name in motors:
        cmd.motor_cmd[go2.LegID[name]].mode = 0x01
        cmd.motor_cmd[go2.LegID[name]].q = target_positions[motor]
        cmd.motor_cmd[go2.LegID[name]].kp = 20.0 
        cmd.motor_cmd[go2.LegID[name]].dq = 0.0
        cmd.motor_cmd[go2.LegID[name]].kd = 0.5
        cmd.motor_cmd[go2.LegID[name]].tau = 0.0

    print("Finished setting motors to initial position.")

def read_joints(msg: LowState_):
    joint_state_log.append((time.time(), msg))

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    rsc = client_utils.get_robot_state_client()
    client_utils.set_service(rsc, "sport_mode", False)

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

    target_positions = {k: SIT_POS[v] for k, v in go2.LegID.items()}

    move_to_initial_pose(cmd, pub, list(go2.LegID.keys()), target_positions)

    import time

    t_end = time.time() + 30
    while time.time() < t_end:
        pass
    
    pub.Close()
    sub.Close()

    print("Closed pub and sub.")

    log_file = "joint_states.pkl"
    with open(log_file, 'wb') as f:
        print(joint_state_log[-1][1].motor_state)
        pickle.dump(joint_state_log, f)
        pickle.dump(joint_command_log, f)

    print(f"Saved to {log_file}")
