import pickle
import numpy as np
import matplotlib.pyplot as plt
import constants.unitree_legged_const as go2
from concurrent.futures import ProcessPoolExecutor

PKL_FILE_AND_MOTORS = {
    "ramp_rate_tests_nov_16/ramp1_FL_0.pkl": ["FL_0"],
    "ramp_rate_tests_nov_16/ramp1_FL_1.pkl": ["FL_1"],
    "ramp_rate_tests_nov_16/ramp1_FR_0.pkl": ["FR_0"],
    "ramp_rate_tests_nov_16/ramp1_FR_1.pkl": ["FR_1"],
    "ramp_rate_tests_nov_16/ramp1_FR_2.pkl": ["FR_2"],
    "ramp_rate_tests_nov_16/ramp1_RL_0.pkl": ["RL_0"],
    "ramp_rate_tests_nov_16/ramp1_RL_1.pkl": ["RL_1"],
    "ramp_rate_tests_nov_16/ramp1_RR_0.pkl": ["RR_0"],
    "ramp_rate_tests_nov_16/ramp1_RR_1.pkl": ["RR_1"],
    "ramp_rate_tests_nov_16/ramp1_RL_2_RR_2_FL_2.pkl": ["RL_2", "RR_2", "FL_2"],
}

def create_default_data():
    return [[], []]

def process_file_wrapper(item):
    return process_file(item[0], item[1])

def process_file(pkl_file, motors):
    """
    Processes a single pickle file and returns data for plotting.
    """
    with open(pkl_file, "rb") as f:
        # Load data once per file
        joint_commands = pickle.load(f)
        joint_states = pickle.load(f)
        info = pickle.load(f)

    # Precompute time offsets and arrays
    time_offset_cmd = joint_commands[0][0]
    joint_command_times = np.array([cmd[0] - time_offset_cmd for cmd in joint_commands])

    time_offset_state = joint_states[0][0]
    joint_state_times = [state[0] - time_offset_state for state in joint_states]

    # Data structures for commands and states
    commands_data = {}
    states_data = {}
    motor_info = {}

    for motor in motors:
        motor_idx = go2.LegID[motor]
        motor_info[motor] = info
        joint_command_values = np.array([cmd[1][motor_idx] for cmd in joint_commands])
        cmd_filter = joint_command_values < 1e6
        joint_command_values = joint_command_values[cmd_filter]

        # Collect motor-specific data
        commands_data[motor] = {
            "times": joint_command_times[cmd_filter],
            "values": joint_command_values,
        }
        states_data[motor] = {
            "times": joint_state_times,
            "values": [state[1][motor_idx].q for state in joint_states],
        }

    return commands_data, states_data, motor_info

if __name__ == "__main__":
    commands_to_plot = {}
    states_to_plot = {}
    motor_infos = {}

    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file_wrapper, PKL_FILE_AND_MOTORS.items())

    # Aggregate results
    for result in results:
        commands_data, states_data, motor_info = result
        for motor in commands_data:
            if motor not in commands_to_plot:
                commands_to_plot[motor] = create_default_data()
            if motor not in states_to_plot:
                states_to_plot[motor] = create_default_data()

            commands_to_plot[motor][0] = commands_data[motor]["times"]
            commands_to_plot[motor][1] = commands_data[motor]["values"]
            states_to_plot[motor][0] = states_data[motor]["times"]
            states_to_plot[motor][1] = states_data[motor]["values"]
            motor_infos[motor] = motor_info[motor]

    # Plot results
    fig, axs = plt.subplots(4,3, figsize=(16, 10), sharex=True)
    fig.suptitle('Joint Commands and Joint States over Time')

    for motor, motor_idx in go2.LegID.items():
        if motor not in motor_infos.keys():
            continue

        axs.flatten()[motor_idx].plot(commands_to_plot[motor][0], commands_to_plot[motor][1], label='Command', color='b')
        axs.flatten()[motor_idx].plot(states_to_plot[motor][0], states_to_plot[motor][1], label='State', color='r', linestyle='--')

        # Plot joint limits
        axs.flatten()[motor_idx].axhline(go2.JOINT_LIMITS[motor][0], color='black', linestyle="dotted")
        axs.flatten()[motor_idx].axhline(go2.JOINT_LIMITS[motor][1], color='black', linestyle="dotted")

        axs.flatten()[motor_idx].set_ylabel(f'{motor} angle (rad)')

    axs.flatten()[-1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
