CXX := g++

SRC := main.cpp
OBJ := main.o

CPPFLAGS := -Iinclude
CXXFLAGS := -Wall -std=gnu++20 -fno-exceptions -m64 -O3 -funroll-loops -mavx -mavx2
LDFLAGS := -s -static -static-libstdc++ -static-libgcc

ifeq ($(OS),Windows_NT)
	EXE := Pingu.exe
	RM := del /q
else
	EXE := Pingu
	RM := rm -f
endif

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

clean:
	$(RM) $(OBJ) $(EXE)
