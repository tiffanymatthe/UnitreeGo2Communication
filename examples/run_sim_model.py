import torch
import sys
import numpy as np
import time
import copy
import pickle

import constants.unitree_legged_const as go2
import examples.actor_critic as actor_critic
import utils.client_utils as client_utils
import utils.state_estimator as se

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

WAIT_LOOPS = True

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
        self.state_estimator = se.StateEstimator()

        self.delayed = False
        self.position_start_delay = None

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

        self.all_cmds = []
        self.all_obs = []

        self.prev_position_target = None
        self.prev_position_target_time = None

        self.policy_every_5_loops = 0

        self.policy_start_time = None

        self.crc = CRC()

        self.joint_limits_in_real_list = list(go2.JOINT_LIMITS.values())

    def start(self):
        '''
        Starts LowCmdWrite thread that infinitely loops
        '''
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=1/self.publisher_frequency, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def load_pt_model(self, model_path):
        '''
        Loads model for policy runs.
        '''
        print(f"Loading model: {model_path}")

        model = actor_critic.ActorCritic(
            num_actor_obs=45,
            num_critic_obs=45,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0
        )

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        model.eval()
        self.model = model

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

        obs = np.concatenate(
            (
                body_ang_vel * normalization.obs_scales.ang_vel,
                grav_vec,
                command * np.array([normalization.obs_scales.lin_vel, normalization.obs_scales.lin_vel, normalization.obs_scales.ang_vel]),
                (self.state_estimator.get_dof_pos_in_sim() - self.default_dof_pos_in_sim) * normalization.obs_scales.dof_pos,
                self.state_estimator.get_dof_vel_in_sim() * normalization.obs_scales.dof_vel,
            )
        )

        if self.policy_output_actions is None:
            self.policy_output_actions = (self.state_estimator.get_dof_pos_in_sim() - self.default_dof_pos_in_sim) / RL_control.action_scale
            self.policy_output_actions = np.clip(self.policy_output_actions, -normalization.clip_actions, normalization.clip_actions)

        obs = np.concatenate((obs, self.policy_output_actions))
        obs = obs.astype(np.float32).reshape(1, -1)

        obs = np.clip(obs, -normalization.clip_observations, normalization.clip_observations)
        self.all_obs.append([time.time(), obs])
        return obs

    def LowCmdWrite(self):
        '''
        Depending on CmdMode, chooses proper cmd motor output to publish and publishes it to the robot.
        '''

        if self.cmd_mode != CmdMode.POLICY:
            self.policy_start_time = time.time()

        if time.time() - self.policy_start_time > 5:
            # print("SWITCHED COMMAND TO FORWARD")
            command = np.array([0.3,0,0])
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
            for i in range(12):
                sim_index = self.state_estimator.joint_idxs_real_to_sim[i]
                self.cmd.motor_cmd[i].q = dof_pos_in_sim[sim_index]
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
            if self.position_percent > 1:
                self.reached_position = True
                print("REACHED POSITION")
                print(self.cmd_mode)
                print(self.state_estimator.allowed_to_run)
            self.position_percent = min(self.position_percent, 1)
            self.prev_position_target = np.zeros(12)
            self.prev_position_target_time = time.time()
            for i in range(12):
                sim_index = self.state_estimator.joint_idxs_real_to_sim[i]
                position_percent = min(self.position_percent, 1)
                self.cmd.motor_cmd[i].q = (1 - position_percent) * self.start_position_in_sim[sim_index] + position_percent * self.target_position_in_sim[sim_index]
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
            # command = np.array([0,0,0]) # np.array([self.state_estimator.cmd_x, self.state_estimator.cmd_y, 0])
            # obs = self.get_observations(command)
            try:
                if self.policy_every_5_loops == 0 or not WAIT_LOOPS:
                    output_actions_in_sim = self.model.actor(torch.from_numpy(obs))
                    output_actions_in_sim = torch.clamp(output_actions_in_sim, -normalization.clip_actions, normalization.clip_actions)
                    self.policy_output_actions = output_actions_in_sim[0].detach().numpy()
                if self.policy_every_5_loops <= 0:
                    self.policy_every_5_loops = 2
                self.policy_every_5_loops -= 1
                self.update_cmd_from_raw_actions(self.policy_output_actions)
            except Exception as e:
                print(f"Inference failed. {e}")
        else:
            raise NotImplementedError(f"{self.cmd_mode} cmd mode not implemented!")
        
        self.cmd.crc = self.crc.Crc(self.cmd)
        if not self.state_estimator.allowed_to_run:
            self.all_cmds.append([time.time(), 3, copy.deepcopy(self.cmd.motor_cmd)])
            self.all_obs[-1].append(3)
        else:
            self.all_cmds.append([time.time(), self.cmd_mode, copy.deepcopy(self.cmd.motor_cmd)])
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

        for i in range(12):
            q = position_targets[self.state_estimator.joint_idxs_real_to_sim[i]]
            q = max(min(q, self.joint_limits_in_real_list[i][1]), self.joint_limits_in_real_list[i][0])
            self.cmd.motor_cmd[i].q = q
            self.cmd.motor_cmd[i].dq = 0
            self.cmd.motor_cmd[i].kp = RL_control.stiffness["joint"]
            self.cmd.motor_cmd[i].kd = RL_control.damping["joint"]
            self.cmd.motor_cmd[i].tau = 0

if __name__ == '__main__':

    # MODIFY:
    # default dof position (used in simulation)

    runner = ModelRunner(publisher_frequency=200)

    model_path = "models/model_bc_rand_env.pt"
    runner.load_pt_model(model_path)
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
