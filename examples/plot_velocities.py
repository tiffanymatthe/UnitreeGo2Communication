import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    with open('/home/fizzer/UnitreeGo2Communication/cmd_est_true_velocities.pkl', 'rb') as f:
        obs = pickle.load(f)
        est_velocities = pickle.load(f)
        true_velocities = pickle.load(f)
    
    est_times = [x[0] for x in est_velocities]
    est_vels = np.array([np.squeeze(np.array(x[1])) for x in est_velocities])

    true_times = [x[1][0] for x in true_velocities if x[1] != (0,0)]
    
    true_vels = np.array([x[1][1] for x in true_velocities if x[1] != (0,0)])
    
    # print(true_vels)
    # print(true_vels)

    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()

    for i in range(3):
        axes[i].plot(est_times, est_vels[:,i], label=f"est {labels[i]}")
        axes[i].plot(true_times, true_vels[:,i], label=f"true {labels[i]}")
        axes[i].legend()
    plt.show()