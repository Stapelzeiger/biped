cmake_minimum_required(VERSION 3.5)
project(biped_robot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes urdf meshes_no_yaw
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
endif()

ament_package()
