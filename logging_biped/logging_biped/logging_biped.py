import rclpy
from rclpy.node import Node
from datetime import datetime
import os
import subprocess
import signal
import sys

class Record(Node):

    def __init__(self):
        super().__init__('record', allow_undeclared_parameters=True)

        self.declare_parameter("test_name", "")
        self.declare_parameter("output_directory", "")
        self.declare_parameter("topics", [""])

        test_name = self.get_parameter("test_name").get_parameter_value().string_value
        self.output_directory = self.get_parameter("output_directory").get_parameter_value().string_value
        self.topics_list = self.get_parameter("topics").get_parameter_value().string_array_value

        self.is_recording = False
        
        now = datetime.now()
        self.time_string = now.strftime("%H-%M-%S")
        date_today = now.strftime("%Y-%m-%d")
        outdir = os.path.join(self.output_directory, date_today)
        self.prefix = '{}_{}'.format(outdir, test_name)
        self.process = None

    def start_recording(self):
        if self.is_recording:
            self.get_logger().info("Recording requested but already recording")
            return

        if self.process:
            return

        self.get_logger().info("Start recording")

        if os.path.exists(self.prefix) is False:
            try:
                self.get_logger().info("Make directory")
                os.mkdir(self.prefix)
            except OSError:
                self.get_logger().warn("Creation of the directory %s failed" % self.prefix)

        
        cmd = ['ros2', 'bag', 'record']

        prefix_bag = os.path.join(self.prefix, 'time_' + self.time_string)
        options = self.topics_list
        options += ['--output={}'.format(prefix_bag)] 
        cmd += options
        self.get_logger().info("Command line: %s" % subprocess.list2cmdline(cmd))
        self.process = subprocess.Popen(' '.join(cmd), shell=True, preexec_fn=os.setsid)

        self.is_recording = True


    def stop_recording(self):
        if not self.is_recording:
            self.get_logger().info("Stop requested but recording not started")
            return

        self.get_logger().info("Stop recording ...")

        if not self.process:
            return

        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        self.process.wait()
        self.process = None

        self.is_recording = False

        self.get_logger().info("Recording stopped!")


def main(args=None):
    rclpy.init(args=args)

    record = Record()
    record.start_recording()

    while True:
        try:
            rclpy.spin(record)
        except KeyboardInterrupt:
            record.stop_recording()
            rclpy.shutdown()

   

    
            


if __name__ == '__main__':
    main()