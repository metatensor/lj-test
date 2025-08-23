cmake_minimum_required(VERSION 3.16)

project(metatomic_lj_test CXX)

# FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
# fixed version in `cmake/FindCUDNN.cmake`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(Torch 1.11 REQUIRED)

add_library(${PROJECT_NAME} SHARED "src/extension.cpp")

target_link_libraries(${PROJECT_NAME} PUBLIC torch)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

# Installation configuration
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)
