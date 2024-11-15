import time
import sys
import math
import pickle
import copy
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
from utils.state_estimator import StateEstimator

PUB_FREQ = 250
PUB_PERIOD = 1/PUB_FREQ

KP = 20
KD = 0.5

crc = CRC()

wrote_to_log_file = False

def get_obs(se: StateEstimator):
    # obs = lin vel, ang vel, projected gravity, commands, dof pos, dof vel, actions
    base_lin_vel = se.body_lin_vel
    base_ang_vel = se.body_ang_vel
    projected_gravity = se.get_gravity_vector()
    commands = ...
    dof_pos = se.get_dof_pos()
    dof_vel = se.get_dof_vel()
    actions = ...

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

    model = ...

    start_time = time.perf_counter()
    period_index = 0

    while True:
        # read from joystick first for termination signal and then go prone
        if stop_program:
            for motor, id in go2.LegID.items():
                # https://github.com/Teddy-Liao/walk-these-ways-go2/blob/ed4cedecfc4f18f4d1cccd1a605cedc5bd111af9/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2/example/state_machine/robot_controller.hpp#L270
                cmd.motor_cmd[id].q = 0.0 # set to init position instead?
                # https://github.com/Teddy-Liao/walk-these-ways-go2/blob/ed4cedecfc4f18f4d1cccd1a605cedc5bd111af9/go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2/example/state_machine/params/params.json#L5
                cmd.motor_cmd[id].dq = 0.0
                cmd.motor_cmd[id].kp = 0
                cmd.motor_cmd[id].kd = KD
                cmd.motor_cmd[id].tau = 0.0

            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            if not wrote_to_log_file:
                log_file = "model_logs.pkl"
                with open(log_file, 'wb') as f:
                    # pickle.dump(pub.logs, f)
                    pickle.dump(joint_state_log, f)
                print(f"Saved to {log_file} after stopping program.")
                wrote_to_log_file = True

            continue

        # latest_joint_state = joint_state_log[-1][1]
        # # TODO: convert joint state to observation that policy can take in
        # obs = ...
        # lin vel, ang vel, projected gravity, commands, dof pos, dof vel, actions

        # action = model(obs)

        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        if elapsed_time >= PUB_PERIOD * (period_index + 1):
            for motor, id in go2.LegID.items():
                if "_0" not in motor:
                    continue
                cmd.motor_cmd[id].q = 0
                cmd.motor_cmd[id].dq = 0.0 # TODO???
                cmd.motor_cmd[id].kp = KP
                cmd.motor_cmd[id].kd = KD
                cmd.motor_cmd[id].tau = 0.0

            # TODO: convert action to a motor cmd that can be sent out
            cmd.crc = crc.Crc(cmd)

            if not pub.Write(cmd):
                print(f"Unable to publish cmd!")

            period_index += 1
        else:
            print(f"Too fast, must wait a bit more.")

        time_to_sleep = PUB_PERIOD * (period_index + 1) - (time.perf_counter() - start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    pub.Close()
    sub.Close()

    log_file = "model_logs.pkl"
    with open(log_file, 'wb') as f:
        pickle.dump(pub.logs, f)
        pickle.dump(joint_state_log, f)

    print(f"Saved to {log_file}")

