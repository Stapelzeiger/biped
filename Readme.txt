# Setup

`git submodule update --init --recursive`

# Dependencies

TODO install script


ros-humble-desktop
ros-humble-robot-state-publisher
ros-humble-compressed-image-transport
ros-humble-xacro

GPIOD:
sudo apt-get install libgpiod-dev

optional
ros-humble-joint-state-publisher-gui
ros-humble-plotjuggler-ros

## Pinocchio
```
git clone https://github.com/stack-of-tasks/pinocchio
cd pinocchio
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DBUILD_PYTHON_INTERFACE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/pinocchio/
make -j4
sudo make install
```

## MuJoCo Viewer

```
pip3 install mujoco
pip3 install mujoco-python-viewer
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
