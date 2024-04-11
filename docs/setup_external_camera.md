
Install spinnaker_camera_driver

```
sudo apt install ros-${ROS_DISTRO}-spinnaker-camera-driver
```

Then run:

```
ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'20538718'"
```