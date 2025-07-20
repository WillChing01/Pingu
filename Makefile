CXX := g++
OBJCOPY := objcopy

SRC := main.cpp
OBJ := main.o

TEST_SRC := test.cpp
TEST_OBJ := test.o

TIME_SRC := process-time-pgn.cpp
TIME_OBJ := process-time-pgn.o

CPPFLAGS := -Iinclude
CXXFLAGS := -Wall -std=gnu++20 -fno-exceptions -m64 -O3 -funroll-loops -mavx -mavx2
LDFLAGS := -s -static -static-libstdc++ -static-libgcc

WEIGHTS_FILES := $(wildcard weights/nnue/*.bin weights/time/*.bin)
WEIGHTS_OBJ := $(WEIGHTS_FILES:.bin=.o)

FILES_TO_CLEAN := $(OBJ) $(TEST_OBJ) $(TIME_OBJ) $(WEIGHTS_OBJ) $(EXE) $(TEST_EXE) $(TIME_EXE)

ifeq ($(OS),Windows_NT)
	OBJCOPY_FLAGS := -I binary -O pei-x86-64 -B i386
	EXE := Pingu.exe
	TEST_EXE := test.exe
	TIME_EXE := process-time-pgn.exe
else
	OBJCOPY_FLAGS := -I binary -O elf64-x86-64 -B i386
	EXE := Pingu
	TEST_EXE := test
	TIME_EXE := process-time-pgn
endif

.PHONY: all test time clean

all: $(EXE)

test: $(TEST_EXE)

time: $(TIME_EXE)

$(EXE): $(OBJ) $(WEIGHTS_OBJ)
	$(CXX) $(OBJ) $(WEIGHTS_OBJ) -o $(EXE) $(LDFLAGS)

$(TEST_EXE): $(TEST_OBJ) $(WEIGHTS_OBJ)
	$(CXX) $(TEST_OBJ) $(WEIGHTS_OBJ) -o $(TEST_EXE) $(LDFLAGS)

$(TIME_EXE): $(TIME_OBJ) $(WEIGHTS_OBJ)
	$(CXX) $(TIME_OBJ) $(WEIGHTS_OBJ) -o $(TIME_EXE) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(TEST_SRC) -o $(TEST_OBJ)

$(TIME_OBJ): $(TIME_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(TIME_SRC) -o $(TIME_OBJ)

$(WEIGHTS_OBJ): %.o: %.bin
	$(OBJCOPY) $(OBJCOPY_FLAGS) $< $@

clean:
ifeq ($(OS),Windows_NT)
	cmd /c "@echo off & for %f in ($(subst /,\,$(FILES_TO_CLEAN))) do if exist "%f" del /q "%f""
else
	@rm -f $(FILES_TO_CLEAN)
endif
