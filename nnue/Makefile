CXX = g++

SRC = parse.cpp
OBJ = parse.o
EXE = parse.exe

SRC2 = dataloader.cpp
OBJ2 = dataloader.o
EXE2 = dataloader.exe

RM = del /q

CPPFLAGS = -Iinclude
CXXFLAGS = -Wall -std=gnu++20 -O3
LDFLAGS = -s -static -static-libstdc++ -static-libgcc

.PHONY: all clean

all: $(EXE) $(EXE2)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LDFLAGS)

$(EXE2): $(OBJ2)
	$(CXX) $(OBJ2) -o $(EXE2) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

$(OBJ2): $(SRC2)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC2) -o $(OBJ2)

clean:
	$(RM) $(OBJ) $(EXE)
	$(RM) $(OBJ2) $(EXE2)
