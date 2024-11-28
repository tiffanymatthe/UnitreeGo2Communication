import pickle

file = "model_logs.pkl"

with open(file, "rb") as f:
    cmds = pickle.load(f)
    motor_states = pickle.load(f)

