cmake_minimum_required(VERSION 3.8)
set(CMAKE_VERBOSE_MAKEFILE on)

project(capture_point)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(biped_bringup REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(include EIGEN3_INCLUDE_DIR)
message(STATUS "EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR}")



add_executable(capture_point src/capture_point.cpp src/foot_trajectory.cpp)
target_include_directories(capture_point PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(capture_point 
                          rclcpp 
                          std_msgs 
                          sensor_msgs 
                          trajectory_msgs 
                          nav_msgs
                          geometry_msgs 
                          biped_bringup
                          tf2
                          tf2_ros
                          visualization_msgs)

install(TARGETS
  capture_point
  DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
