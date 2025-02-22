import pickle
import matplotlib.pyplot as plt
import numpy as np

REAL_JOINT_LABELS = np.array(["FR_0","FR_1","FR_2","FL_0","FL_1","FL_2","RR_0","RR_1","RR_2","RL_0","RL_1","RL_2"])
REAL_TO_SIM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

pkl_file = "data/all_cmds_0_4_diagonal_works.pkl"

with (open(pkl_file, "rb")) as openfile:
    joint_commands = pickle.load(openfile)
    joint_states = pickle.load(openfile)
    print(f"Number of commands {len(joint_commands)} and states {len(joint_states)}")
    # weird, even in policy mode the states we get is less, definitely need to attach time stamps
    # got the damping mode in policy mode, which is why it is hard to tell.
    # print(len([x for x in joint_commands if x[0] == 2]))

# analyze state, clipped

# yaw pitch roll from gyroscope
obs_times = [obs[0] for obs in joint_states]
obs_modes = [obs[2] if len(obs) > 2 else 0 for obs in joint_states]
angular_velocities = [obs[1][0][0:3] for obs in joint_states]
grav_vectors = [obs[1][0][3:6] for obs in joint_states]
lin_x_y_yaw_commands = [obs[1][0][6:9] for obs in joint_states]
dof_positions = [obs[1][0][9:9+12] for obs in joint_states]
dof_velocities = [obs[1][0][9+12:9+24] for obs in joint_states]
# might need to clamp initial one too (yes initial might be way off)
policy_output_actions = [obs[1][0][9+24:9+36] for obs in joint_states]

print(f"Have we analyzed all observations? Size of an observation: {len(joint_states[0][1][0])} vs. last index: {9+36}.")

fig, axs = plt.subplots(4, 2, figsize=(12, 8))

start_index_policy = obs_modes.index(2)
end_index_policy = obs_modes.index(3)

axs[0, 0].plot(obs_times[start_index_policy:end_index_policy], angular_velocities[start_index_policy:end_index_policy])
axs[0, 0].set_title('Angular Velocities')

axs[0, 1].plot(obs_times[start_index_policy:end_index_policy], grav_vectors[start_index_policy:end_index_policy])
axs[0, 1].set_title('Gravitational Vectors')

axs[1, 0].plot(obs_times[start_index_policy:end_index_policy], lin_x_y_yaw_commands[start_index_policy:end_index_policy])
axs[1, 0].set_title('Linear X Y Yaw Commands')

axs[1, 1].plot(obs_times[start_index_policy:end_index_policy], dof_positions[start_index_policy:end_index_policy])
axs[1, 1].set_title('DOF Positions')

axs[2, 0].plot(obs_times[start_index_policy:end_index_policy], dof_velocities[start_index_policy:end_index_policy])
axs[2, 0].set_title('DOF Velocities')

axs[2, 1].plot(obs_times[start_index_policy:end_index_policy], policy_output_actions[start_index_policy:end_index_policy])
axs[2, 1].set_title('Policy Output Actions')

axs[3,0].plot(obs_times[start_index_policy:end_index_policy], obs_modes[start_index_policy:end_index_policy])

# try to correlate current positions and actions
fig1, axs1 = plt.subplots(4, 3, figsize=(12, 8))
axs1 = axs1.flatten()

# try to correlate current positions and actions
fig2, axs2 = plt.subplots(4, 3, figsize=(12, 8))
axs2 = axs2.flatten()

# maybe this was changed in code? need to check
default_pos_in_real = np.array([-0.1, 1.1, -2.0, -0.1, 1.1, -2.0, -0.1, 1.1, -2.6, -0.1, 1.1, -2.6])
default_dof_pos_in_sim = default_pos_in_real[REAL_TO_SIM]

for i in range(12):
    scaled_position = [x[i] + default_dof_pos_in_sim[i] for x in dof_positions][start_index_policy:end_index_policy]
    scaled_action = [x[i] * 0.25 + default_dof_pos_in_sim[i] for x in policy_output_actions][start_index_policy:end_index_policy]
    axs1[i].plot(obs_times[start_index_policy:end_index_policy], scaled_position, label="position (rad)") # use action_scale
    axs1[i].plot(obs_times[start_index_policy:end_index_policy], scaled_action, label="action (rad)")
    slope_of_action = np.gradient(scaled_action, obs_times[start_index_policy:end_index_policy])
    axs2[i].plot(obs_times[start_index_policy:end_index_policy], slope_of_action * 180 / np.pi, label="slope of action (deg/s)")
    # assume it runs at a frequency of 250 Hz (need to save time next time)
    # Modify the data by changing each point relative to the previous point
    # modified_positions = [policy_output_actions[0][i]]  # Start with the initial position
    # max_angle_change = 30 * np.pi / 180 / 250
    # for j in range(1, len(policy_output_actions)):
    #     delta = policy_output_actions[j][i] - policy_output_actions[j-1][i]
    #     if delta > max_angle_change:
    #         delta = max_angle_change
    #     elif delta < -max_angle_change:
    #         delta = -max_angle_change
    #     modified_positions.append(modified_positions[-1] + delta)
    # axs1[i].plot(modified_positions, label="modified action")

    # real_idx = REAL_TO_SIM[i]
    # # assume we stop program only after simulation
    # # seems like joint commands do not change ... they keep the same as go to position
    # # something wrong here...
    # joint_cmd = [cmd[2][real_idx].q for cmd in joint_commands]
    # axs1[i].plot(joint_cmd, label="cmd")
    
    axs1[i].set_title(REAL_JOINT_LABELS[REAL_TO_SIM[i]])
    axs2[i].set_title(REAL_JOINT_LABELS[REAL_TO_SIM[i]])

plt.legend()

plt.tight_layout()
plt.show()