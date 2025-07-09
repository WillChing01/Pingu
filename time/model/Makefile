CXX = g++

SRC = dataloader.cpp
OBJ = dataloader.o
DLL = dataloader.dll

RM = del /q

CXXFLAGS = -Wall -std=gnu++20 -O3
LDFLAGS = -s -static -static-libstdc++ -static-libgcc -shared

.PHONY: all clean

all: $(DLL)

$(DLL): $(OBJ)
	$(CXX) $(OBJ) -o $(DLL) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

clean:
	$(RM) $(OBJ) $(DLL)
