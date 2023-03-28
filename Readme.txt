# Setup

`git submodule update --init --recursive`

# Dependencies

TODO install script


ros-humble-desktop
ros-humble-robot-state-publisher
ros-humble-compressed-image-transport
ros-humble-xacro

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