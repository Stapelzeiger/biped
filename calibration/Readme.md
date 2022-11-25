
# Convert rosbags from ROS2 to ROS1

pip3 install rosbags>=0.9.12 # might need -U
rosbags-convert <ros2_bag_folder> --dst calib_01.bag 

# Camera Calibration

rosrun kalibr kalibr_calibrate_cameras --bag /data/intrinsics_front.bag --models pinhole-radtan pinhole-radtan --topics /camera_front/infra1/image_rect_raw_throttle /camera_front/infra2/image_rect_raw_throttle --target /data/aprilgrid.yaml

rosrun kalibr kalibr_calibrate_imu_camera --bag /data/imu_cam_front.bag --target /data/aprilgrid.yaml --imu /data/imu_vn100.yaml --cams /data/intrinsics_front-camchain.yaml
