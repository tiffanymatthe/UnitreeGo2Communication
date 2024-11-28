import pickle

with open("joint_states.pkl", "rb") as f:
    joint_states = pickle.load(f)
    motor_cmds = pickle.load(f)

print([x.q for x in joint_states[-1][1].motor_state])
print(motor_cmds[-1][1])