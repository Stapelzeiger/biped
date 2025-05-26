# Setup (Ubuntu24.04)

`git submodule update --init --recursive`

# Dependencies

TODO install script

1. `sudo apt install -y ros-jazzy-desktop ros-jazzy-robot-state-publisher ros-jazzy-compressed-image-transport ros-jazzy-xacro`
1. `sudo apt install -y libgpiod-dev`

optional
1. `sudo apt install -y ros-jazzy-joint-state-publisher-gui ros-jazzy-plotjuggler-ros`

## Pinocchio
```
git clone https://github.com/stack-of-tasks/pinocchio.git --branch v3.6.0
ln -s pinocchio path/to/your/ros2_ws/src/pinocchio
```

Note: For faster build, you can disable tests, python bindings, and examples.

# MuJoCo latest (tested on 3.3.0)
```
pip3 --break-system-packages install mujoco
```

## OSQP
```
git clone --recursive https://github.com/osqp/osqp --branch v1.0.0
cd osqp
cmake -B build -G "Unix Makefiles" .
cmake --build build
sudo cmake --build build --target install
```

## OSQP-Eigen
Make sure that Eigen and OSQP are installed, then follow: https://github.com/robotology/osqp-eigen?tab=readme-ov-file#%EF%B8%8F-build-from-source-advanced

Eigen should be installed already in Ubuntu, but incase, run: `sudo apt install libeigen3-dev`
```
git clone https://github.com/robotology/osqp-eigen.git --branch v0.10.0
cd osqp-eigen
cmake -B build -DCMAKE_INSTALL_PREFIX:PATH=/opt/osqp_eigen .
cmake --build build
sudo cmake --build build --target install
```
Then add `export OsqpEigen_DIR="/opt/osqp_eigen"` to your `~/.bashrc` file (or anywhere where you set envjazzyment variables).

## rviz (optional, if you want visualization)
```
sudo apt install ros-jazzy-rviz-2d-overlay-plugins
```

## building the code
make a build.sh with the following
```
#!/bin/bash                                                                                      
                                                                                                 
cd `dirname -- "${BASH_SOURCE[0]}"`                                                              
pwd                                                                                              
                                                                                                 
colcon build --symlink-install --continue-on-error --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo #--event-handlers console_direct+
colcon test --ctest-args test --packages-select capture_point
colcon test-result --all --verbose
```

Then run
```
./build.sh
```

## Motion Capture System

```
cd ros2_ws/src/
git clone --recursive https://github.com/IMRCLab/motion_capture_tracking.git
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


## RL Policy path

Set the policy path in bashrc.
```
export POLICY_PATH={policy_path}
```
