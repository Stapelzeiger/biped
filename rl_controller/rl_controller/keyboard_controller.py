import numpy as np
import mujoco


class KeyboardController:
    def __init__(self):
        self.command = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.max_velocity = 1.0  # m/s
        self.max_yaw_rate = 1.0  # rad/s
        self.velocity_step = 0.1  # m/s per key press
        self.yaw_step = 0.1  # rad/s per key press
        self.running = True

        # Key states
        self.keys_pressed = set()

        print("Keyboard controls:")
        print("  Arrow Up/Down: Forward/Backward velocity")
        print("  Arrow Left/Right: Left/Right velocity")
        print("  A/D: Yaw rotation")
        print("  Space: Stop all movement")
        print("  Q: Quit")

    def keyboard_callback(self, keycode):
        """MuJoCo keyboard callback"""
        if keycode == mujoco.viewer.KEY_UP:
            print("Move forward")
            print('hereee')
            self.command[0] = min(self.command[0] + self.velocity_step, self.max_velocity)
        elif keycode == mujoco.viewer.KEY_DOWN:
            print("Move backward")
            self.command[0] = max(self.command[0] - self.velocity_step, -self.max_velocity)
        elif keycode == mujoco.viewer.KEY_LEFT:
            print("Move left")
            self.command[1] = min(self.command[1] + self.velocity_step, self.max_velocity)
        elif keycode == mujoco.viewer.KEY_RIGHT:
            print("Move right")
            self.command[1] = max(self.command[1] - self.velocity_step, -self.max_velocity)
        elif keycode == ord('a') or keycode == ord('A'):
            print("Yaw left")
            self.command[2] = min(self.command[2] + self.yaw_step, self.max_yaw_rate)
        elif keycode == ord('d') or keycode == ord('D'):
            print("Yaw right")
            self.command[2] = max(self.command[2] - self.yaw_step, -self.max_yaw_rate)
        elif keycode == ord(' '):
            print("Stop movement")
            self.command = np.array([0.0, 0.0, 0.0])
        elif keycode == ord('q') or keycode == ord('Q'):
            print("Quitting...")
            self.running = False

    def update_command(self):
        """Update command with decay when no keys are pressed"""
        # Decay velocities when no keys are pressed
        if abs(self.command[0]) > 0.01:
            self.command[0] *= 0.95
        else:
            self.command[0] = 0.0

        if abs(self.command[1]) > 0.01:
            self.command[1] *= 0.95
        else:
            self.command[1] = 0.0

        if abs(self.command[2]) > 0.01:
            self.command[2] *= 0.95
        else:
            self.command[2] = 0.0

    def get_command(self):
        return self.command.copy()

    def stop(self):
        self.running = False
