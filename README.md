# UnitreeGo2Communication

## Installation

### First-Time Steps (only do once)
We will use a conda environment.

1. Install [conda](https://docs.anaconda.com/miniconda/).

2. Make sure conda is on `$PATH`. Do something like [export path](https://stackoverflow.com/a/35246794).

3. `conda create --name env python=3.8`.

4. Activate environment with `conda activate env`.

4. Install the forked [unitree_sdk2_python](git@github.com:tiffanymatthe/unitree_sdk2_python.git) when environment is activated.

```
cd ~
sudo apt install python3-pip
git clone git@github.com:tiffanymatthe/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

5. Install other packages.

### Every time
`conda activate env` to activate environment.
`conda deactivate` to deactivate environment.

## Telemetry Test

Run all joints through a sinusoidal wave with `joint_pub_sub.py`. This will spawn two threads.

1. Thread 1 will send joint positions to the robot through a publisher at 500 Hz. These commands will be logged.
2. Thread 2 will read joint positions from the robot through a subscriber at 500 Hz. These readings will be logged.