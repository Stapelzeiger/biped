import launch
import time
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

timestr = time.strftime("%Y%m%d-%H-%M-%S")

list_of_topics = ['/joy',
                '/imu',
                '/odometry',
                '/joint_states',
                '/foot_positions',
                '/poll_L_FOOT/gpio',
                '/poll_R_FOOT/gpio',
                '/moteus/motor_sensors',
                '/joint_trajectory',
                '/tf',
                '/tf_static',
                '/capture_point/markers_next_footstep',
                '/capture_point/markers_next_safe_footstep',
                '/capture_point/markers_dcm',
                '/capture_point/markers_desired_dcm',
                '/capture_point/markers_traj_feet',
                '/capture_point/markers_swing_foot_BF',
                '/capture_point/markers_stance_foot_BF',
                '/ik_interface/markers',
                '/rosout',
                '/blackfly_0/image_raw/compressed']


def generate_launch_description():

    test_name = LaunchConfiguration('test_name')
    test_name_launch_arg = DeclareLaunchArgument(
        'test_name',
        default_value='test_name'
    )


    return launch.LaunchDescription([
        test_name_launch_arg,
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'record', *list_of_topics, '--output=/home/sorina/Documents/code/biped_hardware/bags/' + timestr + "_"],
            output='screen'
        )
    ])