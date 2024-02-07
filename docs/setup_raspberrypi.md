### Raspberry Pi

#### Before installation
1. Install Ubuntu 22.04 LTS
1. Install ROS2 Humble

#### Installation
1. Follow the instructions in the README to install the package, download all the required packages

#### Post installation
1. Set static ip
    1. Create a new file: `/etc/netplan/99_config.yaml`
    1. ```
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



