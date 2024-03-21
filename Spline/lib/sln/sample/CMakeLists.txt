file(GLOB hdrs "*.h*" "../../include/*.h")
file(GLOB srcs "*.cpp" "../../samples/sample.cpp")

add_executable(sample ${srcs} ${hdrs})