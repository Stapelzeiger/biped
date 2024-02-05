
### Connecting a motor for testing

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
