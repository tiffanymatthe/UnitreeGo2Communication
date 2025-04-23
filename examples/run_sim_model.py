# python3 -m examples.run_sim_model --command 0.5 0.0 0.0

import torch
import sys
import numpy as np
import time
import copy
import pickle
import configparser
import os

from collections import deque

import constants.unitree_legged_const as go2
import examples.actor_critic as actor_critic
import utils.client_utils as client_utils
import utils.state_estimator as se
from utils.actor import Actor

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

SIM_DT = 0.02 # 1/50 # seconds

LOG = True

HISTORY_LEN = 3
ACTION_HISTORY_LEN = 3

# normal observations + action history + pos and vel history
OBS_SIZE = 36 + 12*ACTION_HISTORY_LEN + 12*HISTORY_LEN*2

class CmdMode:
    NONE = 0
    TO_POSITION = 1
    POLICY = 2
    DAMP = 3

class normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.

class commands:
    class ranges:
        lin_vel_x = [-1.0, 1.0] # min max [m/s]
        lin_vel_y = [-1.0, 1.0]   # min max [m/s]
        ang_vel_yaw = [-1, 1]    # min max [rad/s]

class RL_control:
    # PD Drive parameters:
    control_type = 'P'
    stiffness = {'joint': 20.}  # [N*m/rad]
    damping = {'joint': 0.5}     # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.25
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4

class manual_control:
    # PD Drive parameters:
    control_type = 'P'
    stiffness = {'joint': 40.}  # [N*m/rad]
    damping = {'joint': 5.}     # [N*m*s/rad]

class ModelRunner:
    def __init__(self, publisher_frequency):

        if len(sys.argv)>1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        rsc = client_utils.get_robot_state_client()
        client_utils.set_service(rsc, "sport_mode", False)

        # Create a publisher to publish the data defined in UserData class
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0]=0xFE
        self.cmd.head[1]=0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = go2.PosStopF
            self.cmd.motor_cmd[i].kp = 0
            self.cmd.motor_cmd[i].dq = go2.VelStopF
            self.cmd.motor_cmd[i].kd = 0
            self.cmd.motor_cmd[i].tau = 0

        self.model = None
        self.vel_estimator = None
        self.state_estimator = se.StateEstimator()

        self.delayed = False
        self.position_start_delay = None

        self.buffer_size = int(np.ceil(SIM_DT * publisher_frequency * HISTORY_LEN))
        self.output_action_buffer = deque(maxlen=self.buffer_size)
        self.past_dof_pos_buffer = deque(maxlen=self.buffer_size)
        self.past_dof_vel_buffer = deque(maxlen=self.buffer_size)
        self.dof_time_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)

        self.publisher_frequency = publisher_frequency
        # from go2_config in unitree_rl_gym
        # FR_0,FR_1,FR_2,FL_0,FL_1,FL_2,RR_0,RR_1,RR_2,RL_0,RL_1,RL_2 # mapping in real
        # FL_0,RL_0,FR_0,RR_0,FL_1,RL_1,FR_1,RR_1,FL_2,RL_2,FR_2,RR_2 # mapping ordered in go config
        go2config_to_real = np.array([2,6,10,0,4,8,3,7,11,1,5,9])
        stand_pos_in_go2config = np.array([0.1,0.1,-0.1,-0.1,0.8,1,0.8,1,-1.5,-1.5,-1.5,-1.5])
        stand_pos_in_real = stand_pos_in_go2config[go2config_to_real]
        sit_pos_in_real = np.array([-0.1, 1.1, -2.0, -0.1, 1.1, -2.0, -0.1, 1.1, -2.6, -0.1, 1.1, -2.6])
        # in sim order
        self.stand_pos_in_sim = stand_pos_in_real[self.state_estimator.joint_idxs_real_to_sim]
        self.sit_pos_in_sim = sit_pos_in_real[self.state_estimator.joint_idxs_real_to_sim]
        self.default_dof_pos_in_sim = self.stand_pos_in_sim # self.sit_pos_in_sim # used in unitree_rl_gym for initialization

        self.cmd_mode = CmdMode.NONE
        self.policy_output_actions = None
        self.start_position_in_sim = None
        self.target_position_in_sim = None
        self.reached_position = False
        self.position_percent = 0
        self.duration_s = 5

        self.policy_times = []

        self.all_cmds = []
        self.all_obs = []

        self.vel_cmd = np.array([0,0,0])

        self.prev_position_target = None
        self.prev_position_target_time = None

        self.policy_start_time = None

        self.crc = CRC()

        self.joint_limits_in_real_list = list(go2.JOINT_LIMITS.values())

    def query_closest_output_action(self, query_time):
        closest_time = min(self.time_buffer, key=lambda t: abs(t - query_time))
        closest_index = self.time_buffer.index(closest_time)
        closest_output_action = self.output_action_buffer[closest_index][1]

        return closest_output_action
    

    def query_closest_dofs(self, query_time):
        closest_dof_time = min(self.dof_time_buffer, key=lambda t: abs(t - query_time))
        closest_dof_index = self.dof_time_buffer.index(closest_dof_time)
        closest_dof_pos = self.past_dof_pos_buffer[closest_dof_index][1]
        closest_dof_vel = self.past_dof_vel_buffer[closest_dof_index][1]

        return closest_dof_pos, closest_dof_vel

    def start(self):
        '''
        Starts LowCmdWrite thread that infinitely loops
        '''
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=1/self.publisher_frequency, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def load_pt_model(self, model_path, vel_estimator_path):
        '''
        Loads model for policy runs.
        '''
        print(f"Loading model: {model_path} and velocity estimator {vel_estimator_path}")

        model = actor_critic.ActorCritic(
            num_actor_obs=OBS_SIZE,
            num_critic_obs=OBS_SIZE,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0
        )

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0'))['model_state_dict'])
        model.eval()
        model = model.to(device="cuda:0")
        self.model = model

        vel_estimator = Actor(
            num_actor_obs=OBS_SIZE - 6,
            num_actions=3,
            actor_hidden_dims=[256, 128],
            activation='elu',
            init_noise_std=1.0,
            noise_std_type="scalar"
        )

        vel_estimator.load_state_dict(torch.load(vel_estimator_path, map_location=torch.device('cuda:0'))['model_state_dict'])
        vel_estimator.eval()
        vel_estimator = vel_estimator.to(device="cuda:0")
        self.vel_estimator = vel_estimator

    def go_to_position(self, target_position_in_sim):
        '''
        Sets start and target positions.

        Loops with some delay until the LowCmdWrite thread reaches the target position.
        '''
        self.reached_position = False
        self.position_start_delay = None
        self.delayed = False
        self.position_percent = 0
        self.start_position_in_sim = self.state_estimator.get_dof_pos_in_sim()
        self.target_position_in_sim = target_position_in_sim
        self.cmd_mode = CmdMode.TO_POSITION
        print(f"Starting to go to position in sim: {target_position_in_sim}")
        while not self.reached_position:
            time.sleep(1/self.publisher_frequency/4)
        print(f"Reached position in sim {target_position_in_sim}")

    def run_policy(self):
        '''
        Uses loaded policy to output actions.

        Can be interrupted if we go into DAMP or SIT mode (SIT mode transitions to DAMP mode).
        '''
        runner.state_estimator.run_mode = se.RunMode.NORMAL
        runner.cmd_mode = CmdMode.POLICY
        while True:
            time.sleep(1/self.publisher_frequency / 4)
            if self.state_estimator.run_mode == se.RunMode.DAMP:
                runner.cmd_mode = CmdMode.DAMP
            elif self.state_estimator.run_mode == se.RunMode.SIT:
                self.go_to_position(self.sit_pos_in_sim)
                # force program to go into damp mode afterwards
                self.state_estimator.run_mode = se.RunMode.DAMP
                runner.cmd_mode = CmdMode.DAMP

    def get_observations(self, command):
        """
        From unitree_rl_gym:
            self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions
                                        ),dim=-1)

        Input: command = [lin_vel_x, lin_vel_y, ang_vel_z]

        Updates self.policy_output_actions
        - outputted from policy and clipped
        
        Returns formatted observation object.
        """
        body_ang_vel = self.state_estimator.get_body_angular_vel()
        grav_vec = self.state_estimator.get_gravity_vector()

        # clip commands to be in range
        command[0] = np.clip(command[0], commands.ranges.lin_vel_x[0], commands.ranges.lin_vel_x[1])
        command[1] = np.clip(command[1], commands.ranges.lin_vel_y[0], commands.ranges.lin_vel_y[1])
        command[2] = np.clip(command[2], commands.ranges.ang_vel_yaw[0], commands.ranges.ang_vel_yaw[1])

        current_dof_pos = (self.state_estimator.get_dof_pos_in_sim() - self.default_dof_pos_in_sim) * normalization.obs_scales.dof_pos
        current_dof_vel = self.state_estimator.get_dof_vel_in_sim() * normalization.obs_scales.dof_vel

        obs = np.concatenate(
            (
                body_ang_vel * normalization.obs_scales.ang_vel,
                grav_vec,
                command * np.array([normalization.obs_scales.lin_vel, normalization.obs_scales.lin_vel, normalization.obs_scales.ang_vel]),
                current_dof_pos,
                current_dof_vel,
            )
        )

        # history is ordered oldest to newest

        if not self.output_action_buffer:
            prev_action_history = (self.state_estimator.get_dof_pos_in_sim() - self.default_dof_pos_in_sim) / RL_control.action_scale
            prev_action_history = np.clip(prev_action_history, -normalization.clip_actions, normalization.clip_actions)
            prev_action_history = np.tile(prev_action_history, ACTION_HISTORY_LEN)
        else:
            prev_action_history = np.concatenate(
                [
                    self.query_closest_output_action(time.time() - SIM_DT * (ACTION_HISTORY_LEN - i))
                    for i in range(ACTION_HISTORY_LEN)
                ]
            )

        if not self.past_dof_pos_buffer:
            prev_dof_pos_history = current_dof_pos
            prev_dof_vel_history = current_dof_vel
        else:
            prev_dof_pos_history = np.concatenate(
                [
                    self.query_closest_dofs(time.time() - SIM_DT * (HISTORY_LEN - i))[0]
                    for i in range(HISTORY_LEN)
                ]
            )
            prev_dof_vel_history = np.concatenate(
                [
                    self.query_closest_dofs(time.time() - SIM_DT * (HISTORY_LEN - i))[1]
                    for i in range(HISTORY_LEN)
                ]
            )

        current_time = time.time()

        self.past_dof_pos_buffer.append((current_time, current_dof_pos.copy()))
        self.past_dof_vel_buffer.append((current_time, current_dof_vel.copy()))
        self.dof_time_buffer.append(time.time())

        obs = np.concatenate((obs, prev_action_history, prev_dof_pos_history, prev_dof_vel_history))

        vel_est_obs = torch.cat((
            obs[3:9],
            obs[12:],
        ),dim=-1)

        vel_est_obs = vel_est_obs.astype(np.float32).reshape(1, -1)

        estimated_lin_vel = self.vel_estimator.act_inference(vel_est_obs)

        # possible concern: if vel_estimator runs too long, observation data is outdated
        obs = np.concatenate((
            estimated_lin_vel * self.obs_scales.lin_vel,
            obs
        ))
        
        obs = obs.astype(np.float32).reshape(1, -1)
        obs = np.clip(obs, -normalization.clip_observations, normalization.clip_observations)
        if LOG:
            self.all_obs.append([time.time(), obs])
        return obs

    def LowCmdWrite(self):
        '''
        Depending on CmdMode, chooses proper cmd motor output to publish and publishes it to the robot.
        '''

        if self.cmd_mode != CmdMode.POLICY:
            self.policy_start_time = time.time()

        if time.time() - self.policy_start_time > 5:
            command = self.vel_cmd
        else:
            command = np.array([0,0,0]) # np.array([self.state_estimator.cmd_x, self.state_estimator.cmd_y, 0])
        obs = self.get_observations(command)

        if self.cmd_mode == CmdMode.NONE:
            return
        elif self.cmd_mode == CmdMode.DAMP or not self.state_estimator.allowed_to_run:
            '''
            Sends current motor positions with high damping. Robot should slowly go to floor.
            '''
            dof_pos_in_sim = self.state_estimator.get_dof_pos_in_sim()
            small_cmd = np.zeros(12)
            for i in range(12):
                sim_index = self.state_estimator.joint_idxs_real_to_sim[i]
                small_cmd[i] = dof_pos_in_sim[sim_index]
                self.cmd.motor_cmd[i].q = small_cmd[i]
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kp = 40
                self.cmd.motor_cmd[i].kd = 5
                self.cmd.motor_cmd[i].tau = 0.0
        elif self.cmd_mode == CmdMode.TO_POSITION:
            '''
            We continue trying to reach the target position slowly.
            Interpolats between start and target position.
            Uses higher kp and kd to go from sit mode to stand mode.
            No feedback, so assumes target position is reached after some delay.
            '''
            self.position_percent += 1 / self.duration_s / self.publisher_frequency
            if self.position_percent > 1.5:
                self.reached_position = True
                print("REACHED POSITION")
                print(self.cmd_mode)
                print(self.state_estimator.allowed_to_run)
            self.prev_position_target = np.zeros(12)
            self.prev_position_target_time = time.time()
            small_cmd = np.zeros(12)
            for i in range(12):
                sim_index = self.state_estimator.joint_idxs_real_to_sim[i]
                position_percent = min(self.position_percent, 1)
                small_cmd[i] = (1 - position_percent) * self.start_position_in_sim[sim_index] + position_percent * self.target_position_in_sim[sim_index]
                self.cmd.motor_cmd[i].q = small_cmd[i]
                self.prev_position_target[sim_index] = self.cmd.motor_cmd[i].q
                self.cmd.motor_cmd[i].dq = 0
                self.cmd.motor_cmd[i].kp = manual_control.stiffness["joint"]
                self.cmd.motor_cmd[i].kd = manual_control.damping["joint"]
                self.cmd.motor_cmd[i].tau = 0
            
        elif self.cmd_mode == CmdMode.POLICY:
            '''
            Gets current observations and converts to actions through policy.
            Clips actions as done in sim.
            '''
            self.policy_times.append(time.time())
            try:
                policy_input = torch.from_numpy(obs).to(device="cuda:0")
                with torch.no_grad():
                    output_actions_in_sim = self.model.actor(policy_input).cpu()
                output_actions_in_sim = torch.clamp(output_actions_in_sim, -normalization.clip_actions, normalization.clip_actions)
                self.policy_output_actions = output_actions_in_sim[0].detach().numpy()
                current_time = time.time()
                self.output_action_buffer.append((current_time, self.policy_output_actions.copy()))
                self.time_buffer.append(current_time)
                small_cmd = self.update_cmd_from_raw_actions(self.policy_output_actions)
            except Exception as e:
                print(f"Inference failed. {e}")
        else:
            raise NotImplementedError(f"{self.cmd_mode} cmd mode not implemented!")
        
        self.cmd.crc = self.crc.Crc(self.cmd)
        if LOG:
            if not self.state_estimator.allowed_to_run:
                self.all_cmds.append([time.time(), 3, small_cmd])
                self.all_obs[-1].append(3)
            else:
                self.all_cmds.append([time.time(), self.cmd_mode, small_cmd])
                self.all_obs[-1].append(self.cmd_mode)
        self.pub.Write(self.cmd)

    def limit_change_in_position_target(self, position_targets):
        '''
        Clamps changes in position targets by 90 deg/s.
        
        Updates self.prev_position_target and self.prev_position_target_time.

        Returns clamped position_targets.
        '''
        if self.prev_position_target is not None and self.prev_position_target_time is not None:
            max_angle_change = 90 * np.pi/ 180 * (time.time() - self.prev_position_target_time)
            for i in range(12):
                position_targets[i] = min(max(position_targets[i], self.prev_position_target[i] - max_angle_change), self.prev_position_target[i] + max_angle_change)

        self.prev_position_target = position_targets
        self.prev_position_target_time = time.time()
        return position_targets

    def update_cmd_from_raw_actions(self, output_actions_in_sim):
        '''
        Updates the cmd with the target actions outputted by a policy trained in simulation.
        output_actions_in_sim: action vector of size 12, ordered as in sim.
        Requires scaling by RL_control.action_scale and offset from the default_dof_pos_in_sim.

        Will clamp large changes in actions.

        Returns nothing, updates self.cmd.
        '''
        position_targets = output_actions_in_sim * RL_control.action_scale + self.default_dof_pos_in_sim

        small_cmd = np.zeros(12)

        for i in range(12):
            q = position_targets[self.state_estimator.joint_idxs_real_to_sim[i]]
            q = max(min(q, self.joint_limits_in_real_list[i][1]), self.joint_limits_in_real_list[i][0])
            small_cmd[i] = q
            self.cmd.motor_cmd[i].q = q
            self.cmd.motor_cmd[i].dq = 0
            self.cmd.motor_cmd[i].kp = RL_control.stiffness["joint"]
            self.cmd.motor_cmd[i].kd = RL_control.damping["joint"]
            self.cmd.motor_cmd[i].tau = 0

        return small_cmd
    
    def warm_up_policy(self, num_iterations=5):
        """
        Performs dummy forward passes with the policy model to reduce initial inference latency.
        """
        print(f"Warming up policy model with {num_iterations} iterations...")
        dummy_obs = torch.zeros((1, OBS_SIZE), dtype=torch.float32).to(device="cuda:0")  # Adjust input size as needed
        dummy_vel_obs = dummy_obs = torch.zeros((1, OBS_SIZE - 6), dtype=torch.float32).to(device="cuda:0")
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model.actor(dummy_obs)
                _ = self.vel_estimator.actor(dummy_vel_obs)
        print("Policy model warm-up complete.")

if __name__ == '__main__':

    # MODIFY:
    # default dof position (used in simulation)

    config = configparser.ConfigParser()
    config.read('config.txt')

    # Extract command
    command_str = config.get('DEFAULT', 'command', fallback='0 0 0')
    command = np.array([float(x) for x in command_str.split()])

    # Validate the command values
    if not np.all((-1.0 <= command) & (command <= 1.0)):
        raise ValueError(f"Command values must be in the range [-1, 1]. Received: {command}")

    runner = ModelRunner(publisher_frequency=50)
    runner.vel_cmd = command

    model_path = config.get('DEFAULT', 'model_path', fallback=None)
    vel_est_path = config.get('DEFAULT', 'vel_est_path', fallback=None)

    # Validate the model path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The specified model path does not exist or is not a file: {model_path}")
    
    if not os.path.isfile(vel_est_path):
        raise FileNotFoundError(f"The specified vel estimator model path does not exist or is not a file: {vel_est_path}")

    runner.load_pt_model(model_path, vel_est_path)
    runner.warm_up_policy(num_iterations=5)  # Perform 5 dummy inferences
    runner.start()

    while not runner.state_estimator.allowed_to_run:
        time.sleep(1/runner.publisher_frequency)

    print("STARTING EVERYTHING")

    try:
        runner.go_to_position(runner.stand_pos_in_sim)

        # start model loop
        print(f"Running policy")
        runner.run_policy()
    except KeyboardInterrupt:
        # Calculate average frequency from obs_times
        time_diffs = np.diff(runner.policy_times)
        frequencies = 1 / time_diffs
        average_frequency = np.mean(frequencies)
        print(f"Average frequency: {average_frequency} Hz")
        # Calculate standard deviation of frequency
        std_dev_frequency = np.std(frequencies)
        print(f"Standard deviation of frequency: {std_dev_frequency} Hz")
        if len(time_diffs) > 5:
            print("Time diffs: ")
            print(time_diffs[0:5])

        print("Program interrupted! Saving all_cmds to 'all_cmds.pkl'.")
        with open('all_cmds.pkl', 'wb') as f:
            pickle.dump(runner.all_cmds, f)
        print(f"Saved {len(runner.all_cmds)} commands to 'all_cmds.pkl'.")

    print("Program finished! Saving all_cmds to 'all_cmds_finished.pkl'.")
    with open('all_cmds_finished.pkl', 'wb') as f:
        pickle.dump(runner.all_cmds, f)
        pickle.dump(runner.all_obs, f)
    print(f"Saved {len(runner.all_cmds)} commands to 'all_cmds_finished.pkl'.")

    # use this to publish consistently: example/go2/low_level/go2_stand_example.py
