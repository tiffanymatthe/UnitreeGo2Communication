import pickle
import matplotlib.pyplot as plt
from constants.unitree_legged_const import LegID

if __name__ == "__main__":
    with (open("sinusoidal_logs.pkl", "rb")) as openfile:
        joint_commands = pickle.load(openfile)
        joint_states = pickle.load(openfile)

    motors_to_plot = ["RL_0", "RR_0", "FL_0", "FR_0"]

    joint_command_times = [command[0] for command in joint_commands]
    joint_command_q = [[command[1][LegID[motor_id]].q for motor_id in motors_to_plot] for command in joint_commands]

    joint_state_times = [state[0] for state in joint_states]
    joint_state_q = [[state[1][LegID[motor_id]].q for motor_id in motors_to_plot] for state in joint_states]

    # plotting
    # Plotting
    fig, axs = plt.subplots(len(motors_to_plot), 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Joint Commands and Joint States over Time')

    for joint_idx in range(len(motors_to_plot)):
        # Extract joint command and state for the current joint
        joint_cmd = [cmd[joint_idx] for cmd in joint_command_q]
        joint_state = [state[joint_idx] for state in joint_state_q]
        
        # Plot joint command
        axs[joint_idx].plot(joint_command_times, joint_cmd, label=f'Joint {motors_to_plot[joint_idx]} Command', color='b')
        
        # Plot joint state
        axs[joint_idx].plot(joint_state_times, joint_state, label=f'Joint {motors_to_plot[joint_idx]} State', color='r', linestyle='--')
        
        axs[joint_idx].set_ylabel(f'Joint {motors_to_plot[joint_idx]}')
        axs[joint_idx].legend()

    axs[-1].set_xlabel('Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()