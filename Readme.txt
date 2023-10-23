# Setup

`git submodule update --init --recursive`

# Dependencies

TODO install script

1. `sudo apt install -y ros-humble-desktop ros-humble-robot-state-publisher ros-humble-compressed-image-transport ros-humble-xacro`
1. `sudo apt install -y libgpiod-dev`

GPIOD:
sudo apt-get install libgpiod-dev

optional
1. `sudo apt install -y ros-humble-joint-state-publisher-gui ros-humble-plotjuggler-ros`

## Pinocchio
```
sudo apt install ros-humble-pinocchio
```

## MuJoCo Viewer
```
pip3 install mujoco
pip3 install mujoco-python-viewer
```

## OSQP
```
git clone --recursive https://github.com/osqp/osqp
cd osqp
mkdir build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo cmake --build . --target install
```

## Gazebo
```
sudo apt-get update
sudo apt-get install lsb-release wget gnupg
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```

# Running

Raspberry IP setup:
Ethernet: 10.0.1.2

login: root@biped-raspi.local

check timesync:
```
chronyc sources
```

force time sync:
```
chronyc -a makestep
```