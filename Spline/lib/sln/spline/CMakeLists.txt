set(target "spline")

file(GLOB hdrs "*.h*" "../../include/*.h")
file(GLOB srcs "*.cpp" "../../src/*.cpp")

add_library(${target} STATIC ${srcs} ${hdrs})