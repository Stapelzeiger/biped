cmake_minimum_required(VERSION 3.8)
project(moteus_drv)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_cmake REQUIRED)

INCLUDE(CheckIncludeFiles)
CHECK_INCLUDE_FILES(bcm_host.h HAS_RASPBERRY_PI_HEADER)

if( HAS_RASPBERRY_PI_HEADER )

  add_library(moteus SHARED
    src/moteus.cpp
    lib_external/pi3hat/lib/cpp/mjbots/pi3hat/pi3hat.cc
    )
  target_compile_definitions(moteus
    PRIVATE "COMPOSITION_BUILDING_DLL")
  ament_target_dependencies(moteus
    "rclcpp"
    "rclcpp_components"
    "std_msgs"
    "sensor_msgs"
    "trajectory_msgs"
  )
  rclcpp_components_register_node(moteus PLUGIN "MoteusServo" EXECUTABLE "moteus_node")

  rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

  target_include_directories(moteus PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_external/pi3hat/lib/cpp
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(moteus bcm_host "${cpp_typesupport_target}")
  target_compile_features(moteus PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS
  moteus
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install Python executables
install(PROGRAMS
  scripts/calibrate.py
  DESTINATION lib/${PROJECT_NAME}
)

endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StampedSensors.msg"
  DEPENDENCIES std_msgs
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)


ament_export_dependencies(rosidl_default_runtime)
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
