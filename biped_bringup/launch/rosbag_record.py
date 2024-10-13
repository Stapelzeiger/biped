import launch
import time
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument, RegisterEventHandler
# from launch.event_handlers import OnExecutionComplete


list_of_topics = ['/joy',
                '/imu',
                '/odometry',
                '/vel_estimation/ekf_innovations',
                '/joint_states',
                '/body_trajectories',
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
                '/capture_point/markers_actual_traj_feet',
                '/capture_point/markers_safety_circle',
                '/capture_point/markers_swing_foot_BF',
                '/capture_point/markers_stance_foot_BF',
                '/capture_point/desired_left_contact',
                '/capture_point/desired_right_contact',
                '/ik_interface/markers',
                '/rosout',
                '/flir_camera/image_raw/compressed',
                '/vel_cmd',
                '/e_stop']


def generate_launch_description():

    # test_name = LaunchConfiguration('test_name')
    # test_name_launch_arg = DeclareLaunchArgument(
    #     'test_name',
    #     default_value='test'
    # )
    timestr = time.strftime("%Y%m%d-%H-%M-%S")
    rosbag_dir = '/home/sorina/Documents/code/biped_hardware/bags/'
    rosbag_name = timestr + '.bag'
    rosbag_file = rosbag_dir + rosbag_name

    run_rosbag_record = launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'record', *list_of_topics, '--output='+rosbag_file],
            output='screen'
        )
    create_symlink = launch.actions.ExecuteProcess(
            cmd=['ln', '-s', '-fn', rosbag_file, rosbag_dir + 'latest.bag'],
            output='screen'
        )
    return launch.LaunchDescription([
        # test_name_launch_arg,
        run_rosbag_record,
        create_symlink,
    ])
