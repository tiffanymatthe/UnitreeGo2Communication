import constants.unitree_legged_const as go2
import utils.state_estimator as se
from examples.actor_critic import ActorCritic
import utils.state_estimator as se

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

import torch
import numpy as np
import time
import copy
import pickle
import sys

class normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.

def load_estimator_model(model_path, num_obs, device="cpu"):
    model = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=3, # linear velocity x y z
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation='elu',
            init_noise_std=1.0
        )
    # ignore critic, just used to initialize
    
    model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model

def get_observations(state_estimator, prev_dof_pos_in_sim, default_dof_pos_in_sim):
    """
    From unitree_rl_gym:
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
    
    Returns formatted observation object.
    """
    body_ang_vel = state_estimator.get_body_angular_vel()
    grav_vec = state_estimator.get_gravity_vector()

    obs = np.concatenate(
        (
            body_ang_vel * normalization.obs_scales.ang_vel,
            grav_vec,
            (state_estimator.get_dof_pos_in_sim() - default_dof_pos_in_sim) * normalization.obs_scales.dof_pos,
            state_estimator.get_dof_vel_in_sim() * normalization.obs_scales.dof_vel,
        )
    )
    if prev_dof_pos_in_sim is None:
        prev_dof_pos_in_sim = state_estimator.get_dof_pos_in_sim()
    # this might be an issue
    policy_output_actions = (prev_dof_pos_in_sim - default_dof_pos_in_sim) / 0.25

    policy_output_actions = np.clip(policy_output_actions, -normalization.clip_actions, normalization.clip_actions)

    obs = np.concatenate((obs, policy_output_actions))
    obs = obs.astype(np.float32).reshape(1, -1)

    obs = np.clip(obs, -normalization.clip_observations, normalization.clip_observations)
    return obs

all_true_lin_velocities = []
latest_true_lin_velocity = (0,0)

dof_position_history = None
dof_velocity_history = None

def _load_latest_linear_velocity(msg: SportModeState_):
    global latest_true_lin_velocity
    # print(f"True: {msg.velocity}")
    latest_true_lin_velocity = (time.time(), msg.velocity)

if __name__ == "__main__":

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    HIST_LEN = 6

    model_path = "/home/fizzer/UnitreeGo2Communication/models/walking_estimator_hist_len_6_no_cmd.pt"
    velocity_estimator = load_estimator_model(model_path, num_obs = 48 - 3 - 3 + HIST_LEN * 2 * 12)
    
    state_estimator = se.StateEstimator()

    go2config_to_real = np.array([2,6,10,0,4,8,3,7,11,1,5,9])
    stand_pos_in_go2config = np.array([0.1,0.1,-0.1,-0.1,0.8,1,0.8,1,-1.5,-1.5,-1.5,-1.5])
    stand_pos_in_real = stand_pos_in_go2config[go2config_to_real]
    default_dof_pos_in_sim = stand_pos_in_real[state_estimator.joint_idxs_real_to_sim]
    prev_dof_pos_in_sim = None

    all_obs = []
    all_estimated_lin_velocities = []

    sport_mode_state_subscription = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sport_mode_state_subscription.Init(_load_latest_linear_velocity, 10)

    try:
        while True:
            # get observations and feed into the estimator
            # save data alongside actual linear velocity
            time.sleep(0.02) # matches sim frequency
            obs = get_observations(state_estimator, prev_dof_pos_in_sim, default_dof_pos_in_sim)

            if dof_position_history is None:
                dof_position_history = torch.from_numpy(obs[:,6:6+12]).repeat(1,HIST_LEN)
                dof_velocity_history = torch.from_numpy(obs[:,6+12:6+24]).repeat(1,HIST_LEN)

            obs = torch.cat((torch.from_numpy(obs), dof_position_history, dof_velocity_history), dim=1)

            all_obs.append(copy.deepcopy(obs))
            prev_dof_pos_in_sim = state_estimator.get_dof_pos_in_sim()
            estimated_linear_velocities = velocity_estimator.act_inference(obs)
            print(f"Estimated: {estimated_linear_velocities}")
            all_estimated_lin_velocities.append((time.time(), estimated_linear_velocities))
            all_true_lin_velocities.append((time.time(), latest_true_lin_velocity))

            dof_position_history = torch.cat((dof_position_history, obs[:,6:6+12]), dim=1)
            dof_position_history = dof_position_history[:,12:]
            dof_velocity_history = torch.cat((dof_velocity_history, obs[:,6+12:6+24]), dim=1)
            dof_velocity_history = dof_velocity_history[:,12:]
    except KeyboardInterrupt:
        with open('cmd_est_true_velocities.pkl', 'wb') as f:
            pickle.dump(all_obs, f)
            pickle.dump(all_estimated_lin_velocities, f)
            pickle.dump(all_true_lin_velocities, f)
        print(f"Saved data to 'cmd_est_true_velocities.pkl'.")