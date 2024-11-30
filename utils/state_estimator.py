"""
State Estimator, modified from https://github.com/Teddy-Liao/walk-these-ways-go2/blob/main/go2_gym_deploy/utils/cheetah_state_estimator.py#L52
"""

import time as time
import numpy as np
import math
import constants.unitree_legged_const as go2
from scipy.spatial.transform import Rotation as R

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

# will run normally until either sit or damp is called. damp supersedes sit
class RunMode:
    NORMAL = 0
    SIT = 1
    DAMP = 2

class StateEstimator:
    def __init__(self):
        
        # mapping from real joints to simulation dof pos order
        # actually goes both ways for this particular array
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        self.body_ang_vel = np.zeros(3)
        self.imu_quat = np.zeros(4)

        self.init_time = time.time()
        self.received_first_legdata = False

        self.low_state_subscription = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscription.Init(self._legdata_imu_cb, 10)

        self.rc_command_subscription = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.rc_command_subscription.Init(self._rc_command_cb, 10)

        self.run_mode = RunMode.NORMAL
        self.key_state = [
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

    def get_body_angular_vel(self):
        return self.body_ang_vel

    def get_gravity_vector(self):
        rotation = R.from_quat(self.imu_quat)
        gravity_vec_w = np.array([0.0, 0.0, -1.0])  # Gravity vector in world
        gravity_proj = -1 * rotation.apply(gravity_vec_w)
        return gravity_proj

    def get_dof_pos(self):
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def _legdata_imu_cb(self, msg: LowState_):
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}s after initialization.")

        self.joint_pos = np.array(msg.motor_state.q)
        self.joint_vel = np.array(msg.motor_state.qd)
        self.tau_est = np.array(msg.motor_state.tau_est)

        self.body_ang_vel = np.array([
            msg.imu_state.gyroscope[0],
            msg.imu_state.gyroscope[1],
            msg.imu_state.gyroscope[2]
        ])

        self.imu_quat = np.array([
            msg.imu_state.quaternion[0],
            msg.imu_state.quaternion[1],
            msg.imu_state.quaternion[2],
            msg.imu_state.quaternion[3]
        ])

    def WirelessControllerHandler(self, msg: WirelessController_):
        for i in range(16):
            self.key_state[i][1] = (msg.keys & (1 << i)) >> i

        if self.key_state[9][1] == 1: # key B
            print(f"Damping!")
            self.run_mode = RunMode.DAMP
        elif self.key_state[8][1] == 1: # key A
            print(f"Sitting!")
            self.run_mode = RunMode.SIT

    def close(self):
        self.low_state_subscription.Close()
        self.rc_command_subscription.Close()
