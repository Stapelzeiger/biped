### Raspberry Pi

#### Before installation
1. Install Ubuntu 22.04 LTS
1. Install ROS2 Humble

#### Installation
1. Follow the instructions in the README to install the package, download all the required packages
1. In addition, install `sudo apt install libraspberrypi-dev`, this used to get `moteus_drv` working. If needed uncomment the `if` condition of the CMakeLists.txt in that folder to ensure the moteus node gets installed.
1. Also install `vectornav`, using commit `716660db7f20825336f2ff701cbe1c4da254e738`, https://github.com/dawonn/vectornav/commits/ros2/
    1. Then change in `packet.cpp`:
```
if (*(result + strlen(result) + 1) == '*')
```
with
```
if (result == NULL || (*(result + strlen(result) + 1) == '*'))
```
    1. Then in the `vector_nav.launch` file, add `remappings=[('/vectornav/imu_uncompensated', '/imu')]`
    1. Also change to use `vn_100_200hz.yaml`
<!-- 1. (optional?) Install `https://github.com/ANYbotics/grid_map`, use commit `74333f037cfce321248dfa7b954a815d4f67d79d` in the ros2 branch
1. (this is old) https://github.com/KumarRobotics/imu_vn_100/tree/dashing -->


#### Post installation
1. Set static ip
    1. Create a new file: `/etc/netplan/99_config.yaml`
    1. Set this:
```
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: false
      addresses:
        - 10.0.1.3/24
      optional: true
```
    1. Change `10.0.1.3` to whatever IP you want.
1. Set static hostname
    1. Modify `/etc/hosts` to include: `10.0.1.3 bipedraspi`
    1. Change `10.0.1.3` to the same IP as above and `bipedraspi` can be changed to anything. It is the same that will be used to redirect.

1. Set static hostname for computer that will SSH in. Set it different than the IP above. So, create: `/etc/netplan/99_config.yaml` again:
```
network:
  version: 2
  ethernets:
    [ETHERNET_NAME]:
      dhcp4: false
      addresses:
        - [IP]/24
      optional: true
```
where IP is your IP and ETHERNET_NAME is your ethernet name, can be obtained from `ifconfig`


