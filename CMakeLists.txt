cmake_minimum_required(VERSION 3.1...3.29)
project(neural-network)

set(CMAKE_CXX_COMPILER g++)

set(SOURCES
    src/main.cpp
    src/NeuralNetwork.cpp
)

set(HEADERS
    include/NeuralNetwork.h
)

include_directories(include)

add_executable(neural-network ${SOURCES} ${HEADERS})