# UnitreeGo2Communication

## Installation

### First-Time Steps (only do once)
We will use a conda environment.

1. Install [conda](https://docs.anaconda.com/miniconda/).

2. Make sure conda is on `$PATH`. Do something like [export path](https://stackoverflow.com/a/35246794).

3. `conda create --name env python=3.9`.

4. Activate environment with `conda activate env`.

4. Install [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) with environment activated.

```
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

5. Install other packages.

### Every time
`conda activate env` to activate environment.
`conda deactivate` to deactivate environment.


