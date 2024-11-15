import time as time
import numpy as np
import math
from filterpy.kalman import KalmanFilter

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

# https://github.com/Teddy-Liao/walk-these-ways-go2/blob/main/go2_gym_deploy/utils/cheetah_state_estimator.py#L52

def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot

class StateEstimator:
    def __init__(self):
        
        # reverse legs
        # self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        # 0-FR, 1-FL, 2-RR, 3-RL
        self.contact_idxs = [1, 0, 3, 2]
        self.joint_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        self.world_lin_vel = np.zeros(3)
        self.world_ang_vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.smoothing_ratio = 0.2

        self.contact_state = np.ones(4)

        # Kalman Filter initialization
        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        self.kf.x = np.zeros(3)  # initial velocity estimate (x, y, z)
        self.kf.F = np.eye(3)    # state transition matrix
        self.kf.H = np.eye(3)    # measurement function (direct observation)
        self.kf.P *= 1.0         # initial uncertainty
        self.kf.R = np.eye(3) * 0.1  # measurement noise
        self.kf.Q = np.eye(3) * 0.1  # process noise

        self.init_time = time.time()
        self.received_first_legdata = False

        self.low_state_subscription = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscription.Init(self._legdata_imu_cb, 10)

        self.rc_command_subscription = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.rc_command_subscription.Init(self._rc_command_cb, 10)

        self.body_loc = np.array([0, 0, 0])
        self.body_quat = np.array([0, 0, 0, 1])

        self.stop_program = False

    def get_body_linear_vel(self):
        self.body_lin_vel = np.dot(self.R.T, self.world_lin_vel)
        return self.body_lin_vel
    
    # CHATGPT
    def get_body_linear_vel(self):
        """Estimate body linear velocity using the Kalman Filter."""
        # Rotation matrix to body frame
        body_vel_estimate = np.dot(self.R.T, self.world_lin_vel)

        # Apply Kalman Filter: Assume self.world_lin_vel is derived from accelerometer (prediction)
        delta_time = time.time() - self.timuprev
        self.kf.predict()
        
        # Update the filter if contacts are detected
        observed_velocities = self.get_observed_velocities()
        if len(observed_velocities) > 0:
            mean_observed_velocity = np.mean(observed_velocities, axis=0)
            self.kf.update(mean_observed_velocity)
        else:
            # Without contacts, use the predicted value with high uncertainty
            self.kf.update(body_vel_estimate + np.random.randn(3) * 0.1)

        # Update world velocity from Kalman Filter result
        self.world_lin_vel = self.kf.x  # Kalman filter velocity estimate
        self.body_lin_vel = np.dot(self.R.T, self.world_lin_vel)
        return self.body_lin_vel

    def get_observed_velocities(self):
        """Calculate velocities based on foot contacts."""
        observed_velocities = []
        for i in range(4):
            if self.contact_state[i]:
                # Compute observed velocity for foot i and append
                # (similar to C++ observed velocity calculation)
                observed_velocity = np.zeros(3)  # Replace with actual calculation
                observed_velocities.append(observed_velocity)
        return observed_velocities

    def get_body_angular_vel(self):
        self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (
                    1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

    def get_contact_state(self):
        return self.contact_state[self.contact_idxs]

    def get_rpy(self):
        return self.euler

    def get_dof_pos(self):
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_yaw(self):
        return self.euler[2]

    def get_body_loc(self):
        return np.array(self.body_loc)

    def get_body_quat(self):
        return np.array(self.body_quat)

    def _legdata_imu_cb(self, msg: LowState_):
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}s after initialization.")

        self.joint_pos = np.array(msg.motor_state.q)
        self.joint_vel = np.array(msg.motor_state.qd)
        self.tau_est = np.array(msg.motor_state.tau_est)

        self.euler = np.array(msg.imu_state.rpy)

        self.R = get_rotation_matrix_from_rpy(self.euler)

        # https://swanhub.co/AaronCheng/Instruction_Learning_Robot/blob/main/go1_gym_deploy/unitree_legged_sdk_bin/lcm_position.cpp
        self.contact_state = 1.0 * (np.array(msg.foot_force) > 200)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.imu_state.rpy - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.imu_state.rpy)

    def WirelessControllerHandler(self, msg: WirelessController_):
        if msg.keys == 512:
            print(f"Stop program!")
            self.stop_program = True

    def close(self):
        self.low_state_subscription.Close()
        self.rc_command_subscription.Close()
