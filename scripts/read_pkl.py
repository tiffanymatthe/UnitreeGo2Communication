import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from constants.unitree_legged_const import LegID, JOINT_LIMITS

if __name__ == "__main__":
    pkl_and_motors = {
        "hips.pkl": ["RR_0", "FL_0", "FR_0"],
        "sinusoidal_logs.pkl": ["RL_0"],
        "RL_1.pkl": ["RL_1"],
        "RR_1.pkl": ["RR_1"],
        "FL_1.pkl": ["FL_1"],
        "FR_1.pkl": ["FR_1"],
        "calves.pkl": ["RL_2", "RR_2", "FL_2", "FR_2"],
    }
    motors_to_plot = ["RL_0", "RR_0", "FL_0", "FR_0","RL_1", "RR_1", "FL_1", "FR_1", "RL_2", "RR_2", "FL_2", "FR_2"]

    # key is motor, value is tuple of time and command
    commands_to_plot = defaultdict(lambda: [[], []])
    states_to_plot = defaultdict(lambda: [[], []])

    for pkl_file, motors in pkl_and_motors.items():
        with (open(pkl_file, "rb")) as openfile:
            joint_commands = pickle.load(openfile)
            joint_states = pickle.load(openfile)
            info = pickle.load(openfile)

        time_offset = joint_commands[0][0]
        joint_command_times = [command[0] - time_offset for command in joint_commands]
        for motor in motors:
            commands_to_plot[motor][0] = joint_command_times
        for time_command in joint_commands:
            for motor in motors:
                commands_to_plot[motor][1].append(time_command[1][LegID[motor]])

        time_offset = joint_states[0][0]
        joint_state_times = [state[0] - time_offset for state in joint_states]
        for motor in motors:
            states_to_plot[motor][0] = joint_state_times
        for time_state in joint_states:
            for motor in motors:
                states_to_plot[motor][1].append(time_state[1][LegID[motor]].q)

    # command_time_diffs = np.diff(joint_command_times)
    # print(f"Publisher commands are separated by {np.mean(command_time_diffs)} seconds with variance {np.var(command_time_diffs)}. {1/np.mean(command_time_diffs)} Hz")

    # state_time_diffs = np.diff(joint_state_times)
    # print(f"Subscriber states are separated by {np.mean(state_time_diffs)} seconds with variance {np.var(state_time_diffs)}. {1/np.mean(state_time_diffs)} Hz")

    # Plotting
    fig, axs = plt.subplots(3,4, figsize=(16, 10), sharex=True)
    fig.suptitle('Joint Commands and Joint States over Time')

    from scipy.interpolate import interp1d

    for joint_idx in range(len(motors_to_plot)):
        motor = motors_to_plot[joint_idx]
        # Extract joint command and state for the current joint
        joint_cmd = commands_to_plot[motor][1]
        joint_cmd_time = commands_to_plot[motor][0]
        joint_state = states_to_plot[motor][1]
        joint_state_time = states_to_plot[motor][0]

        print(f"{motor}: Min: {min(joint_state[10:]) * 180 / np.pi} and max: {max(joint_state[10:]) * 180 / np.pi}")

        state_interp_func = interp1d(joint_state_time, joint_state, kind='linear', fill_value='extrapolate')  # Linear interpolation
        state_interp = state_interp_func(joint_cmd_time)  # Resampled state signal

        # Step 2: Perform cross-correlation to assess delay
        correlation = np.correlate(joint_cmd - np.mean(joint_cmd), state_interp - np.mean(state_interp), mode='full')
        delay_idx = np.argmax(correlation) - (len(joint_cmd_time) - 1)
        time_delay = delay_idx * (joint_cmd_time[1] - joint_cmd_time[0])  # Time between samples

        print(f"{motor}: Estimated time delay: {time_delay} seconds")
        
        # Plot joint command
        axs.flatten()[joint_idx].plot(joint_cmd_time, joint_cmd, label=f'Command', color='b')
        
        # Plot joint state
        axs.flatten()[joint_idx].plot(joint_state_time, joint_state, label=f'State', color='r', linestyle='--')

        # plot joint limits
        axs.flatten()[joint_idx].axhline(JOINT_LIMITS[motor][0], color='black', linestyle="dotted")
        axs.flatten()[joint_idx].axhline(JOINT_LIMITS[motor][1], color='black', linestyle="dotted")
        
        axs.flatten()[joint_idx].set_ylabel(f'Joint {motor} angle (rad)')
        axs.flatten()[joint_idx].legend()

    axs.flatten()[-1].set_xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
