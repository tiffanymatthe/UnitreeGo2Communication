import pickle
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

real_file = "data/apr_24/apr_24_back_0_5.pkl"
sim_file = "data/apr_24/Apr23_21-40-49_/Apr23_21-40-49__-0.5_0_0.pickle"
HIST_LEN = 3

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
DOF_POS_OBS_SCALE = 1
ACTION_SCALE = 0.25
go2config_to_real = np.array([2,6,10,0,4,8,3,7,11,1,5,9])
stand_pos_in_go2config = np.array([0.1,0.1,-0.1,-0.1,0.8,1,0.8,1,-1.5,-1.5,-1.5,-1.5])
stand_pos_in_real = stand_pos_in_go2config[go2config_to_real]
DEFAULT_POS_IN_SIM = stand_pos_in_real[REAL_TO_SIM]

[axs, axs1, axs2] = pickle.load(open(sim_file, "rb"))

plt.show()

with (open(real_file, "rb")) as openfile:
    joint_commands = pickle.load(openfile)
    joint_states = pickle.load(openfile)

obs_times = [obs[0] for obs in joint_states]
obs_modes = [obs[2] if len(obs) > 2 else 0 for obs in joint_states]
linear_velocities = [obs[1][0][0:3] for obs in joint_states]
angular_velocities = [obs[1][0][3:6] for obs in joint_states]
grav_vectors = [obs[1][0][6:9] for obs in joint_states]
lin_x_y_yaw_commands = [obs[1][0][9:12] for obs in joint_states]
dof_positions = [obs[1][0][12:12+12] for obs in joint_states]
dof_velocities = [obs[1][0][12+12:12+24] for obs in joint_states]
# might need to clamp initial one too (yes initial might be way off)
policy_output_actions = [obs[1][0][36 + (HIST_LEN - 1) * 12:36 + HIST_LEN * 12] for obs in joint_states]

start_index_policy = next((i for i, x in enumerate(lin_x_y_yaw_commands) if x[0]), None)
# start_index_policy = [x[0] for x in lin_x_y_yaw_commands].index()
try:
    end_index_policy = obs_modes.index(3)
except ValueError:
    end_index_policy = len(obs_modes)

SIM_FREQUENCY = 50
for ax_row in axs:
    for ax in ax_row:
        if ax.lines:  # Check if the axis has any plotted data
            for line in ax.lines:
                x_data, y_data = line.get_data()
                scaled_x_data = [x / SIM_FREQUENCY for x in x_data]
                line.set_data(scaled_x_data, y_data)

time_offset = -obs_times[start_index_policy]
obs_times = [obs_time + time_offset for obs_time in obs_times]

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

# Update the plots to reflect the trimmed data
for ax_row in axs:
    for ax in ax_row:
        if ax.lines:  # Check if the axis has any plotted data
            for line in ax.lines:
                x_data, y_data = line.get_data()
                trimmed_x_data = [x for x in x_data if x <= obs_times[end_index_policy]]
                trimmed_y_data = y_data[:len(trimmed_x_data)]
                line.set_data(trimmed_x_data, trimmed_y_data)

# Adjust the layout to minimize blank space
plt.tight_layout()

# Set x-axis limits to focus on the data range
for ax_row in axs:
    for ax in ax_row:
        if ax.lines:  # Check if the axis has any plotted data
            x_data = [line.get_data()[0] for line in ax.lines]
            if x_data:
                min_x = min([min(data) for data in x_data])
                max_x = max([max(data) for data in x_data])
                ax.set_xlim(min_x, max_x)

subplot_titles = [
    (["gyro_x", "gyro_y", "gyro_z"], "rad/s"),
    (["gravity_x", "gravity_y", "gravity_z"], "m/s^2"),
    (["cmd_vel_x", "cmd_vel_y", "cmd_vel_yaw"], "m/s"),
    (REAL_JOINT_LABELS[REAL_TO_SIM], "rad"),
    (REAL_JOINT_LABELS[REAL_TO_SIM], "rad/s"),
    (REAL_JOINT_LABELS[REAL_TO_SIM], "rad"),
]

# Iterate through each subplot in axs
for row_idx, ax_row in enumerate(axs):
    for col_idx, ax in enumerate(ax_row):
        if ax.lines:  # Check if the axis has any plotted data
            sim_lines = ax.lines[:len(ax.lines)//2]  # Assume first half are sim data
            real_lines = ax.lines[len(ax.lines)//2:]  # Assume second half are real data
            
            # Ensure equal number of sim and real lines
            assert len(sim_lines) == len(real_lines), "Mismatch in number of sim and real data lines."

            # Determine the number of rows needed for three columns
            num_lines = len(sim_lines)
            num_rows = (num_lines + 2) // 3  # Ceiling division for rows

            # Create a new figure for this plot
            fig, comparison_axes = plt.subplots(num_rows, 3, figsize=(20, num_rows * 4))
            fig.suptitle(ax.get_title(), fontsize=16)

            # Flatten the axes array for easier indexing
            comparison_axes = comparison_axes.flatten()

            # Compare each pair of sim and real data lines
            for line_idx, (sim_line, real_line) in enumerate(zip(sim_lines, real_lines)):
                # Extract data
                sim_x, sim_y = sim_line.get_data()
                real_x, real_y = real_line.get_data()

                # Plot on the corresponding subplot
                comparison_axes[line_idx].plot(sim_x, sim_y, label="Sim", color='blue')
                comparison_axes[line_idx].plot(real_x, real_y, label="Real", color='orange')
                comparison_axes[line_idx].legend()
                comparison_axes[line_idx].set_title(subplot_titles[row_idx * 2 + col_idx][0][line_idx])
                comparison_axes[line_idx].set_xlabel("Time (s)")
                comparison_axes[line_idx].set_ylabel(subplot_titles[row_idx * 2 + col_idx][1])

            # Hide any unused subplots
            for unused_ax in comparison_axes[num_lines:]:
                unused_ax.axis('off')

            # Adjust layout for the figure
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title

for ax in axs1:
    if ax.lines:  # Check if the axis has any plotted data
        for line in ax.lines:
            x_data, y_data = line.get_data()
            scaled_x_data = [x / SIM_FREQUENCY for x in x_data]
            line.set_data(scaled_x_data, y_data)
cmd_times = [cmd[0] for cmd in joint_commands]
cmd_modes = [cmd[1] for cmd in joint_commands]
start_index_policy_cmd = cmd_modes.index(2)
try:
    end_index_policy_cmd = cmd_modes.index(3)
except ValueError:
    end_index_policy_cmd = len(cmd_modes)

cmd_times = [cmd_time + time_offset for cmd_time in cmd_times]

for i in range(12):
    scaled_position = [x[i] / DOF_POS_OBS_SCALE + DEFAULT_POS_IN_SIM[i] for x in dof_positions][start_index_policy:end_index_policy]
    scaled_action = [x[i] * ACTION_SCALE + DEFAULT_POS_IN_SIM[i] for x in policy_output_actions][start_index_policy:end_index_policy]
    trimmed_times = np.array(obs_times[start_index_policy:end_index_policy]) - obs_times[start_index_policy]

    axs1[i].plot(trimmed_times, scaled_position, label="position (rad)") # use action_scale

    real_idx = REAL_TO_SIM[i]
    joint_cmd = [cmd[2][real_idx] for cmd in joint_commands]
    trimmed_cmd_times = np.array(cmd_times[start_index_policy_cmd:end_index_policy_cmd]) - obs_times[start_index_policy]

    axs1[i].plot(trimmed_cmd_times, joint_cmd[start_index_policy_cmd:end_index_policy_cmd], label="cmd")

    label = REAL_JOINT_LABELS[REAL_TO_SIM[i]]

    axs1[i].axhline(JOINT_LIMITS[label][0], linestyle="--", color="black")
    axs1[i].axhline(JOINT_LIMITS[label][1], linestyle="--", color="black")
    
    axs1[i].set_title(label)

for ax in axs1:
    if ax.lines:  # Check if the axis has any plotted data
        for line in ax.lines:
            x_data, y_data = line.get_data()
            trimmed_x_data = [x for x in x_data if x <= obs_times[end_index_policy]]
            trimmed_y_data = y_data[:len(trimmed_x_data)]
            line.set_data(trimmed_x_data, trimmed_y_data)

for ax in axs1:
    if ax.lines:  # Check if the axis has any plotted data
        x_data = [line.get_data()[0] for line in ax.lines]
        if x_data:
            min_x = min([min(data) for data in x_data])
            max_x = max([max(data) for data in x_data])
            ax.set_xlim(min_x, max_x)

titles = ["Actual Joint Positions (in sim coordinates)", "Target Positions (output of policy, in sim coordinates)"]

# Create comparison plots for axs1
for line_idx in range(2):  # Assuming there are two lines (sim and real) for each subplot
    # Create a new figure for this comparison with 3 columns and 4 rows
    fig, comparison_axes = plt.subplots(4, 3, figsize=(20, 16))  # Adjust figure size as needed
    fig.suptitle(titles[line_idx], fontsize=16)

    # Flatten the axes array for easier indexing
    comparison_axes = comparison_axes.flatten()

    for i, ax in enumerate(axs1):
        if ax.lines:  # Check if the axis has any plotted data
            # Extract data for the specified line index
            sim_x, sim_y = ax.lines[line_idx].get_data()  # Sim data
            real_x, real_y = ax.lines[line_idx + len(ax.lines) // 2].get_data()  # Real data
            
            # Truncate data if negative real_x
            sim_x, sim_y = zip(*[(x, y) for x, y in zip(sim_x, sim_y) if x >= 0])
            real_x, real_y = zip(*[(x, y) for x, y in zip(real_x, real_y) if x >= 0])

            # Plot on the corresponding subplot
            comparison_axes[i].plot(sim_x, sim_y, label="Sim" if i == 0 else None, color='blue')
            comparison_axes[i].plot(real_x, real_y, label="Real" if i == 0 else None, color='orange')
            comparison_axes[i].legend()
            comparison_axes[i].set_title(REAL_JOINT_LABELS[REAL_TO_SIM[i]])
            comparison_axes[i].set_xlabel("Time (s)")
            comparison_axes[i].set_ylabel("rad")

    # Hide any unused subplots (if there are fewer than 12)
    for j in range(len(axs1), len(comparison_axes)):
        comparison_axes[j].axis('off')

    # Adjust layout for the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title

for ax in axs2:
    if ax.lines:  # Check if the axis has any plotted data
        for line in ax.lines:
            x_data, y_data = line.get_data()
            trimmed_x_data = [x for x in x_data if x <= obs_times[end_index_policy]]
            trimmed_y_data = y_data[:len(trimmed_x_data)]
            line.set_data(trimmed_x_data, trimmed_y_data)

for ax in axs2:
    if ax.lines:  # Check if the axis has any plotted data
        x_data = [line.get_data()[0] for line in ax.lines]
        if x_data:
            min_x = min([min(data) for data in x_data])
            max_x = max([max(data) for data in x_data])
            ax.set_xlim(min_x, max_x)

linear_velocities_labels = ["lin_vel_x", "lin_vel_y", "lin_vel_z"]
for i in range(3):
    lin_vels = [x[i] for x in linear_velocities]
    axs2[i].plot(obs_times[start_index_policy:end_index_policy], lin_vels[start_index_policy:end_index_policy], label="lin vel real", color="red")
    axs2[i].set_title(linear_velocities_labels[i])
    axs2[i].legend()
    

# Show all comparison plots
# Create a directory to save the plots
output_dir = "data/plots_output"
os.makedirs(output_dir, exist_ok=True)

# Save all figures as images
print("saving")
for i, fig in enumerate(plt.get_fignums()):
    plt.figure(fig)
    plt.savefig(os.path.join(output_dir, f"plot_{i + 1}.png"))

print(f"All plots have been saved in the '{output_dir}' folder.")