# UnitreeGo2Communication

## Installation

### First-Time Steps (only do once)
We will use a conda environment.

1. Install [conda](https://docs.anaconda.com/miniconda/).

2. Make sure conda is on `$PATH`. Do something like [export path](https://stackoverflow.com/a/35246794).

3. `conda create --name env python=3.8`.

4. Activate environment with `conda activate env`.

4. Install the forked [unitree_sdk2_python](https://github.com/tiffanymatthe/unitree_sdk2_python) when environment is activated by following the instructions below:

```
cd ~
sudo apt install python3-pip
git clone git@github.com:tiffanymatthe/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

You will need to run `pip3 install -e .` again if you change files in the `unitree_sdk2_python` repository.

Remember to `cd` back to this repository when running scripts.

5. Install other packages. (None for the moment.) [onnx_runtime](https://onnxruntime.ai/docs/get-started/with-python.html)

6. When running the scripts in examples, connect to the robot with an ethernet. First time connection requires setting up the network interface. See [network environment setup](https://support.unitree.com/home/en/developer/Quick_start) for details.

### Every time
`conda activate env` to activate environment.
`conda deactivate` to deactivate environment.

## Telemetry Test

1. `conda activate env`
2. `python3 -m examples.stop_sport_mode enp...` where enp... is the ethernet name.
3. Run all joints through a sinusoidal wave with `python3 -m examples.joint_pub_sub enp...`. This will create a pickle file of subscriber and publisher messages.
4. Read them with `python3 -m scripts.read_pkl`.

