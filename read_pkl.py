import pickle

if __name__ == "__main__":
    with (open("sinusoidal_logs.pkl", "rb")) as openfile:
        print(len(pickle.load(openfile)))
        print(len(pickle.load(openfile)))
