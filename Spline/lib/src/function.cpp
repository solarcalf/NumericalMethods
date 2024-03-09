#include <stdio.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <omp.h>




int f() {
    return 1;
}

extern "C" int F() {
    return f();
}

extern "C" void echo(char* str) {
    printf("%s", str);
}

extern "C" char* alloc_memory() {
    char* str = strdup("Hello!");
    printf("Memory allocated...\n");
    return str;
}

extern "C" void free_memory(char* ptr) {
    printf("Memory deallocated...\n");
    free(ptr);
}

extern "C" int* incArray(int* arr, int size) {

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i)
        arr[i] += 1;
    
    return arr;
}

extern "C" int* getVector() {
    static std::vector<int> v(1000);
    for (size_t i = 0; i < 1000; ++i)
        v[i] = i;
    return &v[0];
}

