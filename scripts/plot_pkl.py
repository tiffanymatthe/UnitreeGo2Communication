import pickle
import matplotlib.pyplot as plt
import numpy as np
JOINT_LIMITS = {
    "FR_0": [-0.837758,0.837758],
    "FR_1": [-1.5708,3.4907],
    "FR_2": [-2.7227, -0.83776],
    "FL_0": [-0.837758,0.837758],
    "FL_1": [-1.5708,3.4907],
    "FL_2": [-2.7227, -0.83776],
    "RR_0": [-0.837758,0.837758],
    "RR_1": [-0.5236,4.5379],
    "RR_2": [-2.7227, -0.83776],
    "RL_0": [-0.837758,0.837758],
    "RL_1": [-0.5236,4.5379],
    "RL_2": [-2.7227, -0.83776],
}

REAL_JOINT_LABELS = np.array(["FR_0","FR_1","FR_2","FL_0","FL_1","FL_2","RR_0","RR_1","RR_2","RL_0","RL_1","RL_2"])
REAL_TO_SIM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
pkl_file = "data/apr_1/cuda_stand.pkl"

DOF_POS_OBS_SCALE = 1
ACTION_SCALE = 0.25
go2config_to_real = np.array([2,6,10,0,4,8,3,7,11,1,5,9])
stand_pos_in_go2config = np.array([0.1,0.1,-0.1,-0.1,0.8,1,0.8,1,-1.5,-1.5,-1.5,-1.5])
stand_pos_in_real = stand_pos_in_go2config[go2config_to_real]
DEFAULT_POS_IN_SIM = stand_pos_in_real[REAL_TO_SIM]
print(DEFAULT_POS_IN_SIM)
print(REAL_JOINT_LABELS[REAL_TO_SIM])

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

# Calculate average frequency from obs_times
time_diffs = np.diff(obs_times)
frequencies = 1 / time_diffs
average_frequency = np.mean(frequencies)
print(f"Average frequency: {average_frequency} Hz")
# Calculate standard deviation of frequency
std_dev_frequency = np.std(frequencies)
print(f"Standard deviation of frequency: {std_dev_frequency} Hz")

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
try:
    end_index_policy = obs_modes.index(3)
except ValueError:
    end_index_policy = len(obs_modes)

time_diffs = np.diff(obs_times[start_index_policy:end_index_policy])
frequencies = 1 / time_diffs
average_frequency = np.mean(frequencies)
print(f"Average frequency of mode 2: {average_frequency} Hz")
std_dev_frequency = np.std(frequencies)
print(f"Standard deviation of frequency of mode 2: {std_dev_frequency} Hz")

print(time_diffs)


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

cmd_times = [cmd[0] for cmd in joint_commands]
cmd_modes = [cmd[1] for cmd in joint_commands]
start_index_policy_cmd = cmd_modes.index(2)
try:
    end_index_policy_cmd = cmd_modes.index(3)
except ValueError:
    end_index_policy_cmd = len(cmd_modes)

for i in range(12):
    scaled_position = [x[i] / DOF_POS_OBS_SCALE + DEFAULT_POS_IN_SIM[i] for x in dof_positions][start_index_policy:end_index_policy]

    scaled_action = [x[i] * ACTION_SCALE + DEFAULT_POS_IN_SIM[i] for x in policy_output_actions][start_index_policy:end_index_policy]

    trimmed_times = np.array(obs_times[start_index_policy:end_index_policy]) - obs_times[start_index_policy]

    axs1[i].plot(trimmed_times, scaled_position, label="position (rad)") # use action_scale
    # axs1[i].plot(trimmed_times, scaled_action, label="action (rad)")
    slope_of_action = np.gradient(scaled_action, trimmed_times)
    axs2[i].plot(trimmed_times, slope_of_action * 180 / np.pi, label="slope of action (deg/s)")

    real_idx = REAL_TO_SIM[i]
    joint_cmd = [cmd[2][real_idx] for cmd in joint_commands]
    trimmed_cmd_times = np.array(cmd_times[start_index_policy_cmd:end_index_policy_cmd]) - obs_times[start_index_policy]

    axs1[i].plot(trimmed_cmd_times, joint_cmd[start_index_policy_cmd:end_index_policy_cmd], label="cmd")

    label = REAL_JOINT_LABELS[REAL_TO_SIM[i]]

    axs1[i].axhline(JOINT_LIMITS[label][0], linestyle="--", color="black")
    axs1[i].axhline(JOINT_LIMITS[label][1], linestyle="--", color="black")
    
    axs1[i].set_title(label)
    axs2[i].set_title(label)
    if i == 11:
        axs1[i].legend()

plt.legend()

plt.tight_layout()
plt.show()
