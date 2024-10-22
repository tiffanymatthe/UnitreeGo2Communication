import pickle
import matplotlib.pyplot as plt
import numpy as np
from constants.unitree_legged_const import LegID

if __name__ == "__main__":
    with (open("joint_2_min_max.pkl", "rb")) as openfile:
        joint_commands = pickle.load(openfile)
        joint_states = pickle.load(openfile)

    motors_to_plot = ["RL_0", "RR_0", "FL_0", "FR_0","RL_1", "RR_1", "FL_1", "FR_1","RL_2", "RR_2", "FL_2", "FR_2"]
    time_offset = joint_commands[0][0]
    joint_command_times = [command[0] - time_offset for command in joint_commands]
    joint_command_q = [[command[1][LegID[motor_id]] for motor_id in motors_to_plot] for command in joint_commands]

    joint_state_times = [state[0] - time_offset for state in joint_states]
    joint_state_q = [[state[1][LegID[motor_id]].q for motor_id in motors_to_plot] for state in joint_states]

    command_time_diffs = np.diff(joint_command_times)
    print(f"Publisher commands are separated by {np.mean(command_time_diffs)} seconds with variance {np.var(command_time_diffs)}. {1/np.mean(command_time_diffs)} Hz")

    state_time_diffs = np.diff(joint_state_times)
    print(f"Subscriber states are separated by {np.mean(state_time_diffs)} seconds with variance {np.var(state_time_diffs)}. {1/np.mean(state_time_diffs)} Hz")

    # plotting
    # Plotting
    fig, axs = plt.subplots(3,4, figsize=(16, 10), sharex=True)
    fig.suptitle('Joint Commands and Joint States over Time')

    for joint_idx in range(len(motors_to_plot)):
        # Extract joint command and state for the current joint
        joint_cmd = [cmd[joint_idx] for cmd in joint_command_q]
        joint_state = [state[joint_idx] for state in joint_state_q]
        
        # Plot joint command
        axs.flatten()[joint_idx].plot(joint_command_times, joint_cmd, label=f'Command', color='b')
        
        # Plot joint state
        axs.flatten()[joint_idx].plot(joint_state_times, joint_state, label=f'State', color='r', linestyle='--')
        
        axs.flatten()[joint_idx].set_ylabel(f'Joint {motors_to_plot[joint_idx]} angle (rad)')
        axs.flatten()[joint_idx].legend()

    axs.flatten()[-1].set_xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
