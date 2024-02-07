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

# Mujoco 3.1
```
pip3 install mujoco
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

## OSQP-Eigen
Make sure that Eigen and OSQP are installed, then follow: https://github.com/robotology/osqp-eigen?tab=readme-ov-file#%EF%B8%8F-build-from-source-advanced

Eigen should be installed already in Ubuntu, but incase, run: `sudo apt install libeigen3-dev`
```
git clone https://github.com/robotology/osqp-eigen.git
cd osqp-eigen
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/osqp_eigen ../
make
make install
```
Then add `export OsqpEigen_DIR="/opt/osqp_eigen"` to your `~/.bashrc` file (or anywhere where you set environment variables).

## rviz (optional, if want visualization)
```
sudo apt install ros-humble-rviz-2d-overlay-plugins
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