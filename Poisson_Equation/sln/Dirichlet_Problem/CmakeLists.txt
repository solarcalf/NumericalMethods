file(GLOB hdrs "*.h*" "../../headers/*.hpp")
file(GLOB srcs "*.cpp" "../../src/test.cpp")

add_executable(Dirichlet_Problem ${srcs} ${hdrs})
