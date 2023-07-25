cmake_minimum_required(VERSION 3.8)
set(CMAKE_VERBOSE_MAKEFILE on)
project(capture_point)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -g)
endif()

option(BUILD_TESTING "Build the testing tree." ON)

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

find_package(OsqpEigen REQUIRED)

add_executable(capture_point src/capture_point.cpp src/trajectory_optimization.cpp)
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
target_link_libraries(capture_point OsqpEigen::OsqpEigen)

add_executable(publisher_test_body_traj src/publisher_test_body_traj.cpp src/foot_trajectory.cpp)
target_include_directories(publisher_test_body_traj PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(publisher_test_body_traj
                          rclcpp
                          std_msgs
                          sensor_msgs
                          trajectory_msgs
                          nav_msgs
                          tf2
                          tf2_ros
                          geometry_msgs)

# add_executable(foot_traj_opt src/trajectory_optimization.cpp)
# target_include_directories(foot_traj_opt PRIVATE ${EIGEN3_INCLUDE_DIR})
# target_link_libraries(foot_traj_opt OsqpEigen::OsqpEigen)

install(TARGETS
  capture_point
  publisher_test_body_traj
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# if(BUILD_TESTING)
#   find_package(ament_cmake_gtest REQUIRED)
#   find_package(OsqpEigen REQUIRED)

#   ament_add_gtest(capture_point_test test/traj_optimization_test.cpp
#                                      src/trajectory_optimization.cpp)

#   target_link_libraries(capture_point_test OsqpEigen::OsqpEigen)
#   target_include_directories(capture_point_test PUBLIC ${EIGEN3_INCLUDE_DIR} include)

#   ament_target_dependencies(capture_point_test rclcpp)
# endif()

set(AMENT_LINT_AUTO OFF)
ament_package()
