import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from pyrpio.i2c import I2C
from pyrpio.i2c_register_device import I2CRegisterDevice
from inductor_sensor_i2c import LDC1612

class InductorSensorPublisher(Node):
    def __init__(self, sensor):
        super().__init__('InductanceSensor')

        self.sensor = sensor

        # Check if the sensor is sleeping.
        if self.sensor.is_sleeping() == True:
            print('Sensor already sleeping. Will config.')
        else:
            print('Set the sensor to sleep.')
            self.sensor.set_sleep_mode(enable=True)

        # Get default config.
        config_value = self.sensor.get_config()
        print("Config (hex)  : 0x{:02X}".format(config_value))
        config_bits = bin(config_value)[2:].zfill(16)
        print(f"Config (bits): {config_bits[:8]} {config_bits[8:]}")

        # Wake up the sensor for reading.
        self.sensor.set_sleep_mode(enable=False)
        print("Sleeping mode: ", self.sensor.is_sleeping())

        self.publisher_ = self.create_publisher(Float32, '/inductor_value', 1)
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        value, err_code = self.sensor.get_channel_data(ch=0)
        max_28bit_value = (1 << 28) - 1
        float_value = value / max_28bit_value

        msg = Float32()
        msg.data = float_value
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    # Open bus.
    i2c1 = I2C('/dev/i2c-1')
    i2c1.open()

    # Initialize sensor.
    sensor_LDC1612 = LDC1612(bus=i2c1)

    node = InductorSensorPublisher(sensor=sensor_LDC1612)
    rclpy.spin(node)
    node.destroy_node()
    i2c1.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()