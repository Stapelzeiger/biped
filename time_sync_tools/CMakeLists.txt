cmake_minimum_required(VERSION 3.8)
project(time_sync_tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)

add_executable(sync_image src/sync_image.cpp)
target_include_directories(sync_image PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(sync_image PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  sync_image
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
  "message_filters"
)

add_executable(vectornav_syncin_event src/vectornav_syncin_event.cpp)
target_include_directories(vectornav_syncin_event PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(vectornav_syncin_event PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  vectornav_syncin_event
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
)


install(TARGETS sync_image vectornav_syncin_event
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
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
