cd src
g++ -mavx2 -mfma -fopenmp -o out.exe test.cpp -I./../headers/ -std=c++17 -O3
./out.exe
rm ./out.exe