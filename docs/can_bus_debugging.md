
### A. Connecting a motor for testing

Interface:

```
┌──────────────────┐     ┌───────────────────┐      ┌─────────────┐
│                  │     │                   │      │             │      ┌────────────┐
│                  │XT90 │                   │ XT60 │             │      │            │
│  Power Supply(1) ├─────►   Power Board(2)  ├──────►   Motor(3)  ◄──────┤  Resistor  │
│                  │     │                   │      │             │      │            │
│                  │     │                   │      │             │      └────────────┘
└──────────────────┘     └────────┬──────────┘      └──────▲──────┘
                                  │                        │
                                  │                        │
                                  │XT60                    │
                                  │                        │
                         ┌────────▼───────────┐            │
                         │                    │            │
                         │    Raspberry Pi    │   JC1-JC5  │
                         │ with CAN Shield(4) ├────────────┘
                         │                    │
                         └────────────────────┘
```
* (1) If you don't spin the motor fast, you can use a standard power supply (i.e., the Pelicase black one). If you spin it fast, then a battery should be used.
* (2) Power Board: https://mjbots.com/products/mjbots-power-dist-r4-5b
* (3) Motor: QDD100 (MjBots)
* (4) https://mjbots.com/products/mjbots-pi3hat-r4-5
* Resistor: https://mjbots.com/products/jst-ph3-can-fd-terminator 


### B. Interface with the CANFD shield 
We use the following CanFD interface: https://www.seeedstudio.com/2-Channel-CAN-BUS-FD-Shield-for-Raspberry-Pi-p-4072.html

Apart from the connections from A., we add the CANFD shield with the Rasperry Pi connection.
The second CAN output of the motor should be connected to one of the CAN inputs on the CANFD shield.

On the Rasperry PI with the CANFD socket, we can setup the CAN interface as follows
```
sudo ip link set can1 up type can tq 25 prop-seg 13 phase-seg1 12 phase-seg2 14 sjw 5   dtq 25 dprop-seg 3 dphase-seg1 1 dphase-seg2 3 dsjw 3   restart-ms 1000 fd on
```

To see the output of CAN:
```
candump can1
```

*Note:* can1 corresponds to 0_L, 0_H, and can0 corresponds to 1_L, 1_H.







