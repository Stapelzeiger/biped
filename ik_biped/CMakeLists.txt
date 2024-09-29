cmake_minimum_required(VERSION 3.8)
set(CMAKE_VERBOSE_MAKEFILE on)

project(ik_biped
  DESCRIPTION "inverse kinematics in Pinocchio"
)

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
find_package(visualization_msgs REQUIRED)
find_package(biped_bringup REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)

find_package(PkgConfig REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(include EIGEN3_INCLUDE_DIR)

message(STATUS "EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR}")
message(STATUS "PINOCCHIO_LINK_LIBRARIES ${PINOCCHIO_LINK_LIBRARIES}")
message(STATUS "pinocchio_LINK_LIBRARIES ${pinocchio_LINK_LIBRARIES}")
message(STATUS "pinocchio_LIBRARIES ${pinocchio_LIBRARIES}")
message(STATUS "PINOCCHIO_LIBRARIES ${PINOCCHIO_LIBRARIES}")
message(STATUS "PINOCCHIO_INCLUDE_DIRS ${PINOCCHIO_INCLUDE_DIRS}")
message(STATUS "PINOCCHIO_FOUND ${PINOCCHIO_FOUND}")
message(STATUS "pinocchio_FOUND ${pinocchio_FOUND}")


add_executable(ik src/ik_interface.cpp src/ik_class_pin.cpp)
target_include_directories(ik PRIVATE ${PINOCCHIO_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(ik
                          rclcpp
                          std_msgs
                          sensor_msgs
                          trajectory_msgs
                          nav_msgs
                          biped_bringup
                          geometry_msgs
                          tf2_ros
                          tf2
                          visualization_msgs)
target_link_libraries(ik ${PINOCCHIO_LIBRARIES})


add_executable(robot_workspace src/robot_workspace.cpp)
target_include_directories(robot_workspace PRIVATE 
                          ${PINOCCHIO_INCLUDE_DIRS}
                          ${EIGEN3_INCLUDE_DIR})

ament_target_dependencies(robot_workspace
                          rclcpp
                          std_msgs
                          visualization_msgs)
target_link_libraries(robot_workspace ${PINOCCHIO_LIBRARIES})



install(TARGETS
  ik
  robot_workspace
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
