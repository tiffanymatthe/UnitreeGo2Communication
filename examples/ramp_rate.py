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
'''

PUB_FREQ = 250
PUB_PERIOD = 1/PUB_FREQ

DEFAULT_RAMP_RATE = 2 * math.pi / 180

DEG_TO_RAD = math.pi / 180

RAMP_RATES = [10 * DEG_TO_RAD,15 * DEG_TO_RAD, 25 * DEG_TO_RAD, 30 * DEG_TO_RAD, 45 * DEG_TO_RAD, 60 * DEG_TO_RAD] #, 90 * DEG_TO_RAD] # rad / s
# RADIAN_SHIFT = [10 * DEG_TO_RAD] #, 20 * DEG_TO_RAD, 30 * DEG_TO_RAD]
MAX_RADIANS = 60 * math.pi / 180

crc = CRC()

joint_state_log = []
joint_command_log = []

def read_joints(msg: LowState_):
    # thread-safety https://stackoverflow.com/a/18568017
    joint_state_log.append((time.time(), msg.motor_state))

def move_to_initial_pose(cmd, pub, motors, target_positions):
    while len(joint_state_log) == 0:
        print(f"Waiting for subscriber to get a data point.")

    period_count = 1

    while True:
        current_time = time.perf_counter()
        for name in motors:
            latest_joint_states = joint_state_log[-1][1]
            if target_positions[name] < latest_joint_states[go2.LegID[name]].q:
                intermediate_target = max(latest_joint_states[go2.LegID[name]].q - DEFAULT_RAMP_RATE * PUB_PERIOD * period_count, target_positions[name])
            else:
                intermediate_target = min(latest_joint_states[go2.LegID[name]].q + DEFAULT_RAMP_RATE * PUB_PERIOD * period_count, target_positions[name])
            cmd.motor_cmd[go2.LegID[name]].mode = 0x01
            cmd.motor_cmd[go2.LegID[name]].q = intermediate_target
            cmd.motor_cmd[go2.LegID[name]].kp = 10.0
            cmd.motor_cmd[go2.LegID[name]].dq = 0.0
            cmd.motor_cmd[go2.LegID[name]].kd = 1.0
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

        latest_joint_states = joint_state_log[-1][1]
        reached_target = True
        for motor in motors:
            if not math.isclose(latest_joint_states[go2.LegID[motor]].q, target_positions[motor], abs_tol=0.15):
                # print(f"Want {latest_joint_states[go2.LegID[motor]].q} to match {target_positions[motor]}")
                reached_target = False
        if reached_target:
            cmd.crc = crc.Crc(cmd)
            if pub.Write(cmd):
                joint_command_log.append((time.time(), [x.q for x in cmd.motor_cmd]))
            else:
                print("Waiting for subscriber.")
            break
    
    for name in motors:
        cmd.motor_cmd[go2.LegID[name]].mode = 0x01
        cmd.motor_cmd[go2.LegID[name]].q = target_positions[motor]
        cmd.motor_cmd[go2.LegID[name]].kp = 10.0 
        cmd.motor_cmd[go2.LegID[name]].dq = 0.0
        cmd.motor_cmd[go2.LegID[name]].kd = 1.0
        cmd.motor_cmd[go2.LegID[name]].tau = 0.0

    print("Finished setting motors to initial position.")

if __name__ == '__main__':

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

    target_positions = {}
    for motor, limits in go2.JOINT_LIMITS.items():
        target_positions[motor] = (limits[0] + limits[1]) / 2

    hip_motors = ["RL_0", "RR_0", "FL_0", "FR_0"]
    move_to_initial_pose(cmd, pub, hip_motors, target_positions)

    # motors_to_control = ["RL_2"] #, "RR_2", "FL_2", "FR_2"]
    # motors_to_control = ["RL_1"] #, "RR_1", "FL_1", "FR_1"]
    motors_to_control = ["RR_0"] #, "RR_0", "FL_0", "FR_0"]

    for motor in motors_to_control:
        limits = go2.JOINT_LIMITS[motor]
        print(f"Changing {target_positions[motor]}")
        target_positions[motor] = limits[0] + 20 * DEG_TO_RAD
        print(f"to {target_positions[motor]}")

    # sets first position to q=0
    move_to_initial_pose(cmd, pub, motors_to_control, target_positions)

    print(f"Finished moving to target positions for initialization.")

    ramp_index = 0
    period_index = 0

    start_time = time.perf_counter()

    print(f"Starting test.")

    while True:
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        if elapsed_time >= PUB_PERIOD * (period_index + 1):
            for motor in motors_to_control:
                change_in_q = RAMP_RATES[ramp_index] * PUB_PERIOD * period_index
                cmd.motor_cmd[go2.LegID[motor]].q = min(target_positions[motor] + change_in_q, go2.JOINT_LIMITS[motor][1])
                # print(f"Sending {min(target_positions[motor] + change_in_q, go2.JOINT_LIMITS[motor][1])}")

            cmd.crc = crc.Crc(cmd)

            # Publish message
            if pub.Write(cmd):
                joint_command_log.append((time.time(), [x.q for x in cmd.motor_cmd]))
            else:
                print("Waiting for subscriber.")

            latest_joint_states = joint_state_log[-1][1]
            reached_target = True
            for motor in motors_to_control:
                if latest_joint_states[go2.LegID[motor]].q < min(MAX_RADIANS + target_positions[motor], go2.JOINT_LIMITS[motor][1]):
                    reached_target = False
                    # print(f"{latest_joint_states[go2.LegID[motor]].q} < {MAX_RADIANS + target_positions[motor]}, {go2.JOINT_LIMITS[motor][1]}")
            if reached_target:
                ramp_index += 1
                period_index = -1
                print(f"Changing shift.")
                move_to_initial_pose(cmd, pub, motors_to_control, target_positions)
                if ramp_index >= len(RAMP_RATES):
                    break
            
            period_index += 1
        # Update next_time and sleep for the remaining time
        time_to_sleep = PUB_PERIOD * (period_index + 1) - (time.perf_counter() - start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    pub.Close()
    sub.Close()

    print("Closed pub and sub.")

    log_file = "ramp1_" + "_".join(motors_to_control) + ".pkl"
    with open(log_file, 'wb') as f:
        pickle.dump(joint_command_log, f)
        pickle.dump(joint_state_log, f)
        pickle.dump({"ramp_rates": RAMP_RATES, "pub_freq": PUB_FREQ, "max_radians": MAX_RADIANS},f)

    print(f"Saved to {log_file}")
