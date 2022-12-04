cmake_minimum_required(VERSION 3.8)
project(linux_gpio)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(time_ref src/time_ref.cpp)
target_include_directories(time_ref PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(time_ref PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  time_ref
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
)
target_link_libraries(
  time_ref
  gpiodcxx
)

add_executable(trigger src/trigger.cpp)
target_include_directories(trigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(trigger PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  trigger
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
)
target_link_libraries(
  trigger
  gpiod
)


install(TARGETS time_ref trigger
  DESTINATION lib/${PROJECT_NAME})

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
