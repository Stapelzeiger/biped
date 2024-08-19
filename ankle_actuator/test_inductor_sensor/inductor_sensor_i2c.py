from pyrpio.i2c import I2C
from pyrpio.i2c_register_device import I2CRegisterDevice
from typing import Tuple, Optional

class LDC1612:
    LDC1612_DATA_BASE = 0x00
    LDC1612_STATUS = 0x18
    LDC1612_CONFIG = 0x1A
    LDC1612_DEVICE_ID = 0x7F
    
    def __init__(self, bus: I2C, address=0x2A):
        self.address = address
        self.i2c_reg = I2CRegisterDevice(bus, address, register_size=1, data_size=2)
        device_id = self.get_device_id()
        assert device_id == 0x3055

    def get_register(self, register: int, mask: Optional[int] = None) -> int:
        value = self.i2c_reg.read_register(register)
        if mask is not None:
            value = value & mask
        return value

    def get_register_bit(self, register: int, bit: int) -> bool:
        mask = 1 << bit
        value = self.get_register(register, mask)
        return bool(value >> bit)

    def set_register(self, register, value: int, mask: Optional[int] = None):
        if mask is not None:
            pvalue = self.get_register(register, ~mask)  # pylint: disable=invalid-unary-operand-type
            value = pvalue | (value & mask)
        self.i2c_reg.write_register(register, value)

    def set_register_bit(self, register: int, bit: int, on: bool):
        mask = 1 << bit
        pvalue = self.get_register(register, ~mask)
        self.set_register(register, pvalue | (int(on) << bit))

    def get_status_errors(self):
        value = self.get_register(self.LDC1612_STATUS) >> 8
        return value

    def get_device_id(self):
        return self.get_register(self.LDC1612_DEVICE_ID)

    def get_config(self):
        return self.get_register(self.LDC1612_CONFIG)

    def get_channel_data(self, ch: int) -> Tuple[int, int]:
        ch_msb = self.get_register(self.LDC1612_DATA_BASE + 2 * ch) # Note: according to 7.5.3, it is ok to read them sequencially.
        #_LSB register must be read after _MSB to ensure data coherency.
        ch_lsb = self.get_register(self.LDC1612_DATA_BASE + 2 * ch + 1)

        value = ((0x0FFF & ch_msb) << 16) | ch_lsb
        err_code = (ch_msb & 0xF000) >> 12

        if value == 0x0FFFFFFF:
            print("Can't detect coil inductance!!!")
            return -1, 0

        return value, err_code

    def set_sleep_mode(self, enable: bool):
        self.set_register_bit(self.LDC1612_CONFIG, 13, enable)

    def is_sleeping(self) -> bool:
        return bool(self.get_register_bit(self.LDC1612_CONFIG, 13))


def main():
    # Open bus.
    i2c1 = I2C('/dev/i2c-1')
    i2c1.open()

    # Initialize sensor.
    sensor_LDC1612 = LDC1612(bus=i2c1)

    # Check if the sensor is sleeping.
    if sensor_LDC1612.is_sleeping() == True:
        print('Sensor already sleeping. Will config.')
    else:
        print('Set the sensor to sleep.')
        sensor_LDC1612.set_sleep_mode(enable=True)

    # Get default config.
    config_value = sensor_LDC1612.get_config()
    print("Config (hex)  : 0x{:02X}".format(config_value))
    config_bits = bin(config_value)[2:].zfill(16)
    print(f"Config (bits): {config_bits[:8]} {config_bits[8:]}")

    # Set the initial configuration.
    # TODO if the default configuration needs adjustments.

    # Wake up the sensor for reading.
    sensor_LDC1612.set_sleep_mode(enable=False)
    print("Sleeping mode: ", sensor_LDC1612.is_sleeping())

    while (1):
        value, err_code = sensor_LDC1612.get_channel_data(ch=0)
        max_28bit_value = (1 << 28) - 1
        float_value = value / max_28bit_value
        print(f' Value [H] {float_value}')

    i2c1.close()

if __name__ == '__main__':
    main()