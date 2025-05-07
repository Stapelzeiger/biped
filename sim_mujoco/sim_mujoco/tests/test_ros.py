#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import mujoco
import mujoco.viewer

class Simulation(Node):
    def __init__(self):
        super().__init__('Simulation')
        self.model = mujoco.MjModel.from_xml_path('../../../biped_robot_description/urdf/custom_robot_v2.mujoco.xml')
        self.data = mujoco.MjData(self.model)

        self.Ts = self.model.opt.timestep

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.timer = self.create_timer(self.Ts, self.timer_callback)

    def timer_callback(self):
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def is_running(self):
        return self.viewer.is_running()

    def stop(self):
        self.viewer.close()

def main(args=None):
    rclpy.init(args=args)

    sim = Simulation()
    while(sim.is_running()):
        rclpy.spin_once(sim)

    sim.stop()
    sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()