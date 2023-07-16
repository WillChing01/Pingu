cmake_minimum_required(VERSION 3.0)
project(release)
set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp)

add_executable(Chess_3 ${SOURCE_FILES})

# g++.exe -Wall -std=gnu++17 -m64 -fexceptions -O3 -std=gnu++17 -Iinclude -c C:\Users\willi\OneDrive\Desktop\Coding\Chess_3\main.cpp -o obj\Release\main.o
# g++.exe  -o bin\Release\Chess_3.exe obj\Release\main.o  -static-libstdc++ -static-libgcc -static -m64 -s