cmake_minimum_required(VERSION 3.8)
project(kinematic_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(biped_bringup REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(pinocchio REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

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


add_executable(vel_estimation src/vel_estimation.cpp)
target_include_directories(vel_estimation PRIVATE ${PINOCCHIO_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

target_compile_features(vel_estimation PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  vel_estimation
  rclcpp
  std_msgs
  sensor_msgs
  biped_bringup
  tf2
  tf2_ros
  nav_msgs
  visualization_msgs
  )

target_link_libraries(vel_estimation ${PINOCCHIO_LIBRARIES})


install(TARGETS vel_estimation
  DESTINATION lib/${PROJECT_NAME})


install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
