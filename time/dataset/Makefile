CXX = g++

SRC = parse.cpp
OBJ = parse.o
EXE = parse.exe

RM = del /q

CXXFLAGS = -Wall -std=gnu++20 -O3
LDFLAGS = -s -static -static-libstdc++ -static-libgcc

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

clean:
	$(RM) $(OBJ) $(EXE)
