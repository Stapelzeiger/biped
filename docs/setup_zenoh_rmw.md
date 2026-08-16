# RMW Zenoh

Based on https://github.com/ros2/rmw_zenoh

## Before installation:
Install rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

## Installation
In your biped_ws/src/,
```
git clone https://github.com/ros2/rmw_zenoh.git
cd ../../ #takes you back to ws
rosdep install --from-paths src --ignore-src --rosdistro iron -y
source /opt/ros/iron/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select rmw_zenoh_cpp zenoh_c_vendor
```
Note we usually build with debug info

## Testing:
first terminal:
```
source install/setup.bash
ros2 run rmw_zenoh_cpp rmw_zenohd # zenoh router
```

second terminal:
```
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run demo_nodes_cpp talker
```

third terminal:
```
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run demo_nodes_cpp listener
```

Note: if we are trying to make a PC talk to raspberry pi, we should take note of this: https://github.com/ros2/rmw_zenoh?tab=readme-ov-file#connecting-multiple-hosts

are connecting 