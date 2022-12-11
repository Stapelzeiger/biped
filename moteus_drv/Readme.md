# Moteus Servo ROS Driver

for realtime priority add the following to `/etc/security/limits.conf`
```
ubuntu           -       rtprio          95
```


# Moteus config notes

aux2
    pin: 0 & 1 
        mode i2c
    spi
        mode disabled
    i2c
        devices 0
            as5048

motor position
    sources: 0
        leve unchanged
    sources: 1
        aux_number 2
        type i2c
        device 0
        cpr 65536
        reference output
        low pass 10Hz
        sign as needed
    output
        reference source 1
    rotor_to_output ratio 0.2 (0.2 hip yaw, 0.166666 hip forward and side, 0.1111111 knee)

remove limits (put to nan)

to save zero oputput:
d cfg-set-output 0
conf write

