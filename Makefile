CXX := g++
OBJCOPY := objcopy

SRC := main.cpp
OBJ := main.o

CPPFLAGS := -Iinclude
CXXFLAGS := -Wall -std=gnu++20 -fno-exceptions -m64 -O3 -funroll-loops -mavx -mavx2
LDFLAGS := -s -static -static-libstdc++ -static-libgcc

WEIGHTS_FILES := $(wildcard weights/nnue/*.bin weights/time/*.bin)
WEIGHTS_OBJ := $(WEIGHTS_FILES:.bin=.o)

ifeq ($(OS),Windows_NT)
	OBJCOPY_FLAGS := -I binary -O pei-x86-64 -B i386
	EXE := Pingu.exe
	RM = $(foreach f,$1,if exist "$(f)" del /q "$(f)" & ) rem
else
	OBJCOPY_FLAGS := -I binary -O elf64-x86-64 -B i386
	EXE := Pingu
	RM := rm -f
endif

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ) $(WEIGHTS_OBJ)
	$(CXX) $(OBJ) $(WEIGHTS_OBJ) -o $(EXE) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

$(WEIGHTS_OBJ): %.o: %.bin
	$(OBJCOPY) $(OBJCOPY_FLAGS) $< $@

clean:
	$(call RM,$(OBJ) $(WEIGHTS_OBJ) $(EXE))
