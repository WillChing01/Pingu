cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 20)

# gcc compiler options
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(GCC_COMPILE_DEBUG_OPTIONS "-Wall;-std=gnu++20;-fexceptions;-pg")
set(GCC_COMPILE_RELEASE_OPTIONS "-Wall;-std=gnu++20;-fexceptions;-funroll-loops;-O3")

set(GCC_LINK_DEBUG_OPTIONS "-static-libstdc++;-static-libgcc;-static;-lgmon;-no-pie;-pg")
set(GCC_LINK_RELEASE_OPTIONS "-static-libstdc++;-static-libgcc;-static;-s")

# include files

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/)
include_directories(${INCLUDE_DIR})

set(SOURCE_FILES main.cpp)

set(HEADER_FILES
    ${INCLUDE_DIR}/bitboard.h
    ${INCLUDE_DIR}/board.h
    ${INCLUDE_DIR}/constants.h
    ${INCLUDE_DIR}/engine-commands.h
    ${INCLUDE_DIR}/evaluation.h
    ${INCLUDE_DIR}/format.h
    ${INCLUDE_DIR}/king.h
    ${INCLUDE_DIR}/knight.h
    ${INCLUDE_DIR}/magic.h
    ${INCLUDE_DIR}/pawn.h
    ${INCLUDE_DIR}/search.h
    ${INCLUDE_DIR}/slider.h
    ${INCLUDE_DIR}/testing.h
    ${INCLUDE_DIR}/transposition.h
    ${INCLUDE_DIR}/tuning.h
    ${INCLUDE_DIR}/uci.h
    ${INCLUDE_DIR}/weights.h
)

# create the project

project(Pingu)

add_executable(Pingu ${SOURCE_FILES} ${HEADER_FILES})
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${GCC_COMPILE_DEBUG_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${GCC_COMPILE_RELEASE_OPTIONS}>")
target_link_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${GCC_LINK_DEBUG_OPTIONS}>")
target_link_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${GCC_LINK_RELEASE_OPTIONS}>")

project(tuning)

add_executable(tuning ${SOURCE_FILES} ${HEADER_FILES})
target_compile_definitions(${PROJECT_NAME} PUBLIC TUNING=1)
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${GCC_COMPILE_DEBUG_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${GCC_COMPILE_RELEASE_OPTIONS}>")
target_link_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${GCC_LINK_DEBUG_OPTIONS}>")
target_link_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${GCC_LINK_RELEASE_OPTIONS}>")
