import torch
import sys
import numpy as np
import time

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
    TO_POSITION = 0
    POLICY = 1
    DAMP = 2
    NONE = 3

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
        # in real joint order
        self.sit_pos = np.array([-0.1, 1.1, -2.0, -0.1, 1.1, -2.0, -0.1, 1.1, -2.6, -0.1, 1.1, -2.6])
        # in sim order
        self.sit_pos = self.sit_pos[self.state_estimator.joint_idxs]
        self.default_dof_pos = self.sit_pos # used in unitree_rl_gym for initialization
        self.cmd_mode = CmdMode.NONE
        self.raw_actions = None
        self.start_position = None
        self.target_position = None
        self.reached_position = False
        self.position_percent = 0
        self.duration_s = 1

        self.joint_limits_list = list(go2.JOINT_LIMITS.values())

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=1/self.publisher_frequency, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def load_pt_model(self, model_path):
        print(f"Loading model: {model_path}")

        model = actor_critic.ActorCritic(
            num_actor_obs=48,
            num_critic_obs=48,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0
        )

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        model.eval()
        self.model = model

    def go_to_position(self, target_position):
        self.reached_position = False
        self.position_percent = 0
        self.state_estimator.cmd_mode = CmdMode.TO_POSITION
        self.start_position = self.state_estimator.get_dof_pos()
        self.target_position = target_position
        while not self.reached_position:
            time.sleep(1/self.publisher_frequency)

    def run_policy(self):
        runner.state_estimator.run_mode = se.RunMode.NORMAL
        runner.state_estimator.cmd_mode = CmdMode.POLICY
        while True:
            time.sleep(1/self.publisher_frequency / 4)
            if self.state_estimator.run_mode == se.RunMode.DAMP:
                runner.state_estimator.cmd_mode = CmdMode.DAMP
            elif self.state_estimator.run_mode == se.RunMode.SIT:
                self.go_to_position(self.sit_pos)
                # force program to go into damp mode afterwards
                self.state_estimator.run_mode = se.RunMode.DAMP
                runner.state_estimator.cmd_mode = CmdMode.DAMP

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
                self.state_estimator.get_body_angular_vel() * normalization.obs_scales.ang_vel,
                self.state_estimator.get_gravity_vector(),
                (self.state_estimator.get_dof_pos() - self.default_dof_pos) * normalization.obs_scales.dof_pos,
                self.state_estimator.get_dof_vel() * normalization.obs_scales.dof_vel,
            )
        )

        if self.raw_actions is None:
            # check ordering
            self.raw_actions = (self.state_estimator.get_dof_pos() - self.default_dof_pos) / control.action_scale

        obs = np.concatenate((obs, self.raw_actions))
        obs = obs.astype(np.float32).reshape(1, -1)

        obs = torch.clip(obs, -normalization.clip_observations, normalization.clip_observations)
        return obs

    def LowCmdWrite(self):
        if self.state_estimator.run_mode == CmdMode.NONE:
            return
        elif self.state_estimator.run_mode == CmdMode.DAMP:
            dof_pos = self.state_estimator.get_dof_pos()
            for i in range(12):
                self.cmd.motor_cmd[i].q = dof_pos[i]
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kp = 40
                self.cmd.motor_cmd[i].kd = 5
                self.cmd.motor_cmd[i].tau = 0.0
        elif self.state_estimator.run_mode == CmdMode.TO_POSITION:
            self.position_percent += 1 / self.duration_s / self.publisher_frequency
            self.position_percent = min(self.position_percent, 1)
            for i in range(12):
                self.cmd.motor_cmd[i].q = (1 - self.position_percent) * self.start_position[i] + self.position_percent * self.target_position[i]
                self.cmd.motor_cmd[i].dq = 0
                self.cmd.motor_cmd[i].kp = self.Kp
                self.cmd.motor_cmd[i].kd = self.Kd
                self.cmd.motor_cmd[i].tau = 0
        elif self.state_estimator.run_mode == CmdMode.POLICY:
            obs = self.get_observations()
            try:
                output_actions = self.model.actor(torch.from_numpy(obs))
                output_actions = torch.clip(output_actions, -normalization.clip_actions, normalization.clip_actions)
                self.raw_actions = output_actions[0].detach().numpy()
            except Exception as e:
                print(f"Inference failed. {e}")

            self.update_cmd_from_raw_actions(output_actions)
        else:
            raise NotImplementedError(f"{self.state_estimator.run_mode} run mode not implemented!")

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def update_cmd_from_raw_actions(self, output_actions):
        position_targets = output_actions * control.action_scale + self.default_dof_pos
        for i in range(12):
            q = position_targets[self.state_estimator.joint_idxs[i]]
            q = max(min(q, self.joint_limits_list[i][1]), self.joint_limits_list[i][0])
            self.cmd.motor_cmd[i].q = q
            self.cmd.motor_cmd[i].dq = 0
            self.cmd.motor_cmd[i].kp = self.Kp
            self.cmd.motor_cmd[i].kd = self.Kd
            self.cmd.motor_cmd[i].tau = 0

if __name__ == '__main__':

    runner = ModelRunner(publisher_frequency=250)

    model_path = "models/model_1500.pt"
    runner.load_pt_model(model_path)

    runner.go_to_position(runner.sit_pos)

    # start model loop
    runner.run_policy()

    # use this to publish consistently: example/go2/low_level/go2_stand_example.py