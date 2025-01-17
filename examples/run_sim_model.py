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

class control:
    # PD Drive parameters:
    control_type = 'P'
    stiffness = {'joint': 20.}  # [N*m/rad]
    damping = {'joint': 0.5}     # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.25
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4

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

        self.Kp = control.stiffness["joint"]
        self.Kd = control.damping["joint"]
        self.publisher_frequency = publisher_frequency
        # from go2_config in unitree_rl_gym
        stand_pos_in_real = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1, -1.5])
        # in real joint order, from eppl-erau-db/go2_rl_ws repo
        sit_pos_in_real = np.array([-0.1, 1.1, -2.0, -0.1, 1.1, -2.0, -0.1, 1.1, -2.6, -0.1, 1.1, -2.6])
        # in sim order
        self.sit_pos_in_sim = sit_pos_in_real[self.state_estimator.joint_idxs_real_to_sim]
        self.stand_pos_in_sim = stand_pos_in_real[self.state_estimator.joint_idxs_real_to_sim]
        self.default_dof_pos_in_sim = self.stand_pos_in_sim # self.sit_pos_in_sim # used in unitree_rl_gym for initialization
        self.cmd_mode = CmdMode.NONE
        self.raw_actions_in_sim = None
        self.start_position_in_sim = None
        self.target_position_in_sim = None
        self.reached_position = False
        self.position_percent = 0
        self.duration_s = 5

        self.all_cmds = []
        self.all_obs = []

        self.prev_position_target = None
        self.prev_position_target_time = None

        self.crc = CRC()

        self.joint_limits_in_real_list = list(go2.JOINT_LIMITS.values())

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=1/self.publisher_frequency, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def load_pt_model(self, model_path):
        print(f"Loading model: {model_path}")

        model = actor_critic.ActorCritic(
            num_actor_obs=42,
            num_critic_obs=42,
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
        self.reached_position = False
        self.position_percent = 0
        self.start_position_in_sim = self.state_estimator.get_dof_pos_in_sim()
        print(self.start_position_in_sim)
        self.target_position_in_sim = target_position_in_sim
        self.cmd_mode = CmdMode.TO_POSITION
        print(f"Starting to go to position in sim: {target_position_in_sim}")
        while not self.reached_position:
            time.sleep(1/self.publisher_frequency/4)
        print(f"Reached position in sim {target_position_in_sim}")

    def run_policy(self):
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

    def get_observations(self):
        """
        From unitree_rl_gym:
            self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions
                                        ),dim=-1)
        """
        obs = np.concatenate(
            (
                self.state_estimator.get_body_angular_vel() * 0, # normalization.obs_scales.ang_vel,
                self.state_estimator.get_gravity_vector() * 0,
                (self.state_estimator.get_dof_pos_in_sim() - self.default_dof_pos_in_sim) * normalization.obs_scales.dof_pos,
                self.state_estimator.get_dof_vel_in_sim() * normalization.obs_scales.dof_vel,
            )
        )

        if self.raw_actions_in_sim is None:
            # check ordering
            self.raw_actions_in_sim = (self.state_estimator.get_dof_pos_in_sim() - self.default_dof_pos_in_sim) / control.action_scale

        obs = np.concatenate((obs, self.raw_actions_in_sim))
        obs = obs.astype(np.float32).reshape(1, -1)

        obs = np.clip(obs, -normalization.clip_observations, normalization.clip_observations)
        self.all_obs.append(obs)
        return obs

    def LowCmdWrite(self):
        if self.cmd_mode == CmdMode.NONE:
            return
        elif self.cmd_mode == CmdMode.DAMP:
            dof_pos_in_sim = self.state_estimator.get_dof_pos_in_sim()
            for i in range(12):
                sim_index = self.state_estimator.joint_idxs_real_to_sim[i]
                self.cmd.motor_cmd[i].q = dof_pos_in_sim[sim_index]
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kp = 40
                self.cmd.motor_cmd[i].kd = 5
                self.cmd.motor_cmd[i].tau = 0.0
        elif self.cmd_mode == CmdMode.TO_POSITION:
            self.position_percent += 1 / self.duration_s / self.publisher_frequency
            if self.position_percent > 1.5:
                self.reached_position = True
            # self.position_percent = min(self.position_percent, 1)
            self.prev_position_target = np.zeros(12)
            self.prev_position_target_time = time.time()
            for i in range(12):
                sim_index = self.state_estimator.joint_idxs_real_to_sim[i]
                position_percent = min(self.position_percent, 1)
                self.cmd.motor_cmd[i].q = (1 - position_percent) * self.start_position_in_sim[sim_index] + position_percent * self.target_position_in_sim[sim_index]
                self.prev_position_target[sim_index] = self.cmd.motor_cmd[i].q
                self.cmd.motor_cmd[i].dq = 0
                self.cmd.motor_cmd[i].kp = self.Kp
                self.cmd.motor_cmd[i].kd = self.Kd
                self.cmd.motor_cmd[i].tau = 0
        elif self.cmd_mode == CmdMode.POLICY:
            obs = self.get_observations()
            try:
                output_actions_in_sim = self.model.actor(torch.from_numpy(obs))
                output_actions_in_sim = torch.clamp(output_actions_in_sim, -normalization.clip_actions, normalization.clip_actions)
                self.raw_actions_in_sim = output_actions_in_sim[0].detach().numpy()
                self.update_cmd_from_raw_actions(self.raw_actions_in_sim)
            except Exception as e:
                print(f"Inference failed. {e}")
        else:
            raise NotImplementedError(f"{self.cmd_mode} cmd mode not implemented!")

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.all_cmds.append((self.cmd_mode, copy.copy(self.cmd.motor_cmd)))
        if self.cmd_mode != CmdMode.POLICY:
            self.pub.Write(self.cmd)

    def limit_change_in_position_target(self, position_targets):
        print(f"Before limiting joint changes: {position_targets}")
        if self.prev_position_target is not None and self.prev_position_target_time is not None:
            max_angle_change = 10 * (time.time() - self.prev_position_target_time)
            for i in range(12):
                position_targets[i] = min(max(position_targets[i], self.prev_position_target[i] - max_angle_change), self.prev_position_target[i] + max_angle_change)
        print(f"After limiting joint changes: {position_targets}")

        self.prev_position_target = position_targets
        self.prev_position_target_time = time.time()
        return position_targets

    def update_cmd_from_raw_actions(self, output_actions_in_sim):
        position_targets = output_actions_in_sim * control.action_scale + self.default_dof_pos_in_sim
        position_targets = self.limit_change_in_position_target(position_targets)

        for i in range(12):
            q = position_targets[self.state_estimator.joint_idxs_real_to_sim[i]]
            q = max(min(q, self.joint_limits_in_real_list[i][1]), self.joint_limits_in_real_list[i][0])
            self.cmd.motor_cmd[i].q = q
            self.cmd.motor_cmd[i].dq = 0
            self.cmd.motor_cmd[i].kp = self.Kp
            self.cmd.motor_cmd[i].kd = self.Kd
            self.cmd.motor_cmd[i].tau = 0

if __name__ == '__main__':

    runner = ModelRunner(publisher_frequency=250)

    model_path = "models/model_500.pt"
    runner.load_pt_model(model_path)
    runner.start()

    try:
        runner.go_to_position(runner.sit_pos_in_sim)

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