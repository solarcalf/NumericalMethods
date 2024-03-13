cmake_minimum_required(VERSION 2.8)

project(spline)

include_directories(../include ../src)

file(GLOB hdrs "*.h*" "../include/*.h")
file(GLOB srcs "*.cpp" "../src/sample.cpp")

# BUILD

add_executable(sample ${srcs} ${hdrs})