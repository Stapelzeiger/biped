import launch
import time
import os

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
                '/capture_point/markers_swing_foot_BF',
                '/capture_point/markers_stance_foot_BF',
                '/ik_interface/markers',
                '/rosout',
                '/external_cam/image_raw/compressed',
                '/e_stop']

def generate_launch_description():
    ROSBAG_NAME = '20240827-16-11-27'
    ROSBAG_DIR = '/home/sorina/Documents/code/biped_hardware/bags/'

    rosbag_file = os.path.join(ROSBAG_DIR, ROSBAG_NAME + '.bag')
    csv_output_dir = os.path.join(ROSBAG_DIR, ROSBAG_NAME + '_csvs/')
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
    
    run_rosbag_record = launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play', rosbag_file],
            output='screen'
        )

    # List of commands to echo topics to CSV.
    echo_commands = [
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'topic', 'echo', '--csv', topic, '>', f'{csv_output_dir}{topic.replace("/", "_")}.csv'],
            shell=True,
            output='screen'
        )
        for topic in list_of_topics
    ]

    return launch.LaunchDescription([
        run_rosbag_record,
        *echo_commands,
    ])