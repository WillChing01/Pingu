CXX = g++

DIR = build
SRC = main.cpp
OBJ = main.o

CPPFLAGS = -Iinclude
CXXFLAGS = -Wall -std=gnu++20 -fno-exceptions -m64 -O3 -funroll-loops -mavx -mavx2
LDFLAGS = -s -static -static-libstdc++ -static-libgcc

ifeq ($(OS),Windows_NT)
	EXE = Pingu.exe
	DEL = del $(DIR)\$(OBJ) $(DIR)\$(EXE)
else
	EXE = Pingu
	DEL = $(RM) $(DIR)/$(OBJ) $(DIR)/$(EXE)
endif

.PHONY: all clean

all: $(DIR)/$(EXE)

$(DIR)/$(EXE): $(DIR)/$(OBJ)
	$(CXX) $(DIR)/$(OBJ) -o $(DIR)/$(EXE) $(LDFLAGS)

$(DIR)/$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(DIR)/$(OBJ)

clean:
	$(DEL)
