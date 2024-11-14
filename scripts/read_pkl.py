import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from constants.unitree_legged_const import LegID, JOINT_LIMITS

def get_largest_slope(A, f):
    f = np.array(f)
    return np.round(2 * np.pi * f * A, 2)

def frequency_stats(times):
    # Calculate time intervals (periods)
    periods = np.diff(times)
    
    # Calculate frequencies (reciprocal of periods)
    frequencies = 1 / periods
    
    # Mean and standard deviation of frequencies
    mean_freq = np.mean(frequencies)
    std_dev_freq = np.std(frequencies)
    
    return mean_freq, std_dev_freq

# Function to find robust crossings within a tolerance range
def find_crossings(y_data, x_data, level):
    crossings = []
    for i in range(2, len(y_data) - 2):
        # Check if the first two points are below the level and the next two points are ascending above the level
        if (y_data[i-2] >= level and y_data[i-1] >= level and
            y_data[i] <= level and y_data[i+1] <= y_data[i]):
            # Register the crossing point as the midpoint between i-1 and i
            slope = (y_data[i] - y_data[i-1]) / (x_data[i] - x_data[i-1])
            crossing_x = x_data[i-1] + (level - y_data[i-1]) / slope
            crossings.append(crossing_x)
    return crossings

def get_lag(y_sin, x_sin, y_exp, x_exp, offset):
    y_sin = np.array(y_sin)
    x_sin = np.array(x_sin)
    y_exp = np.array(y_exp)
    x_exp = np.array(x_exp)

    sin_interp = interp1d(x_sin, y_sin, kind='cubic', fill_value="extrapolate")
    y_sin_interp = sin_interp(x_exp)

    # Detect crossings in both sinusoidal and experimental data
    sin_crossings = find_crossings(y_sin_interp, x_exp, offset)
    exp_crossings = find_crossings(y_exp, x_exp, offset)

    if len(exp_crossings) > 0 and exp_crossings[0] < 3:
        exp_crossings = np.delete(exp_crossings, 0)

    # Calculate lag for each crossing, ensuring the indexes match in length
    if len(sin_crossings) != len(exp_crossings):
        print(f"Double check {sin_crossings}, {exp_crossings}")
    min_crossings = min(len(sin_crossings), len(exp_crossings))
    lags = []
    for i in range(min_crossings):
        sin_x_cross = sin_crossings[i]
        exp_x_cross = exp_crossings[i]
        lag = exp_x_cross - sin_x_cross
        lags.append(lag)

    # Convert lags list to numpy array for easy manipulation
    lags = np.array(lags)
    return lags, sin_crossings, exp_crossings

def remove_spikes(y_sin):
    # Detect and remove spikes in y_sin by replacing them with average of neighbors
    threshold = 0.5  # Define a threshold for spike detection
    for i in range(1, len(y_sin) - 1):
        if abs(y_sin[i] - y_sin[i-1]) > threshold and abs(y_sin[i] - y_sin[i+1]) > threshold:
            y_sin[i] = (y_sin[i-1] + y_sin[i+1]) / 2  # Replace spike with average of neighbors

    return y_sin

if __name__ == "__main__":
    # pkl_and_motors = {
    #     "hips.pkl": ["RR_0", "FL_0", "FR_0"],
    #     "sinusoidal_logs.pkl": ["RL_0"],
    #     "RL_1.pkl": ["RL_1"],
    #     "RR_1.pkl": ["RR_1"],
    #     "FL_1.pkl": ["FL_1"],
    #     "FR_1.pkl": ["FR_1"],
    #     "calves.pkl": ["RL_2", "RR_2", "FL_2", "FR_2"],
    # }
    # motors_to_plot = ["RL_0", "RR_0", "FL_0", "FR_0","RL_1", "RR_1", "FL_1", "FR_1", "RL_2", "RR_2", "FL_2", "FR_2"]
    motors_to_plot = ["RL_0", "FL_0", "FR_0","RL_1", "RR_1", "FL_1", "FR_1", "RL_2", "RR_2", "FR_2"]

    # key is motor, value is tuple of time and command
    commands_to_plot = defaultdict(lambda: [[], []])
    states_to_plot = defaultdict(lambda: [[], []])
    offsets = {}
    ramp_rates = {}

    for motor in motors_to_plot: # pkl_file, motors in pkl_and_motors.items():
        motors = [motor]
        # pkl_file=f"ramp1_FR_0.pkl"
        pkl_file=f"ramp1_{motor}.pkl"
        with (open(pkl_file, "rb")) as openfile:
            joint_commands = pickle.load(openfile)
            joint_states = pickle.load(openfile)
            info = pickle.load(openfile)
            # print(f"{motor}: {get_largest_slope(info['amplitude'][motor], info['freq'])}")
            # ramp_rate = get_largest_slope(info['amplitude'][motor], info['freq'])
            # ramp_rates[motor] = ramp_rate
            # offsets[motor] = info["offset"][motor]

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

    for motor in motors_to_plot:
        joint_idx = LegID[motor]
        # motor = motors_to_plot[joint_idx]
        # Extract joint command and state for the current joint
        joint_cmd = commands_to_plot[motor][1]
        joint_cmd_time = commands_to_plot[motor][0]
        joint_state = states_to_plot[motor][1]
        joint_state_time = states_to_plot[motor][0]

        # print(f"{motor}: Average publisher frequency: {frequency_stats(joint_cmd_time)}")
        # print(f"{motor}: Average subscriber frequency: {frequency_stats(joint_state_time)}")

        # print(f"{motor}: Min: {min(joint_state[10:]) * 180 / np.pi} and max: {max(joint_state[10:]) * 180 / np.pi}")

        state_interp_func = interp1d(joint_state_time, joint_state, kind='linear', fill_value='extrapolate')  # Linear interpolation
        state_interp = state_interp_func(joint_cmd_time)  # Resampled state signal

        # Step 2: Perform cross-correlation to assess delay
        correlation = np.correlate(joint_cmd - np.mean(joint_cmd), state_interp - np.mean(state_interp), mode='full')
        delay_idx = np.argmax(correlation) - (len(joint_cmd_time) - 1)
        time_delay = delay_idx * (joint_cmd_time[1] - joint_cmd_time[0])  # Time between samples

        # print(f"{motor}: Estimated time delay: {time_delay} seconds")

        joint_cmd = remove_spikes(joint_cmd)

        # lags, cmd_crossings, state_crossings = get_lag(joint_cmd, joint_cmd_time, joint_state, joint_state_time, offsets[motor])
        # print(f"{motor}: lag times {lags}")

        # print(f"{motor}: {ramp_rates[motor]} and {lags}")

        # axs.flatten()[joint_idx].scatter(ramp_rates[motor], lags)
        # axs.flatten()[joint_idx].set_ylabel(f'Joint {motor} lag (s)')

        if motor == "RL_0":
            print(joint_state)
        
        # Plot joint command
        axs.flatten()[joint_idx].plot(joint_cmd_time, joint_cmd, label=f'Command', color='b')
        # axs.flatten()[joint_idx].scatter(cmd_crossings, np.ones(len(cmd_crossings)) * offsets[motor], s=15)
        
        # Plot joint state
        axs.flatten()[joint_idx].plot(joint_state_time, joint_state, label=f'State', color='r', linestyle='--')
        # axs.flatten()[joint_idx].scatter(state_crossings, np.ones(len(state_crossings)) * offsets[motor], s=15)

        # plot joint limits
        axs.flatten()[joint_idx].axhline(JOINT_LIMITS[motor][0], color='black', linestyle="dotted")
        axs.flatten()[joint_idx].axhline(JOINT_LIMITS[motor][1], color='black', linestyle="dotted")
        
        axs.flatten()[joint_idx].set_ylabel(f'Joint {motor} angle (rad)')
    
    axs.flatten()[-1].legend()
    axs.flatten()[-1].set_xlabel("Ramp Rate (rad/s)")
    # axs.flatten()[-1].set_xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
