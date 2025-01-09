import pickle

with open('all_cmds.pkl', 'rb') as f:
    all_cmds = pickle.load(f)
print(f"Loaded {len(all_cmds)} commands.")
print(all_cmds)
