CXX := g++
OBJCOPY := objcopy

SRC := main.cpp
OBJ := main.o

CPPFLAGS := -Iinclude
CXXFLAGS := -Wall -std=gnu++20 -fno-exceptions -m64 -O3 -funroll-loops -mavx -mavx2
LDFLAGS := -s -static -static-libstdc++ -static-libgcc

WEIGHTS_FILES := $(wildcard weights/nnue/*.bin)
WEIGHTS_OBJ := $(WEIGHTS_FILES:.bin=.o)

ifeq ($(OS),Windows_NT)
	OBJCOPY_FLAGS := -I binary -O pei-x86-64 -B i386 --set-section-alignment .data=32
	EXE := Pingu.exe
else
	OBJCOPY_FLAGS := -I binary -O elf64-x86-64 -B i386 --set-section-alignment .data=32
	EXE := Pingu
endif

FILES_TO_CLEAN := $(OBJ) $(WEIGHTS_OBJ) $(EXE)

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ) $(WEIGHTS_OBJ)
	$(CXX) $(OBJ) $(WEIGHTS_OBJ) -o $(EXE) $(LDFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

$(WEIGHTS_OBJ): %.o: %.bin
	@$(OBJCOPY) $(OBJCOPY_FLAGS) $< $@

clean:
ifeq ($(OS),Windows_NT)
	@cmd /c "@echo off & for %f in ($(subst /,\,$(FILES_TO_CLEAN))) do if exist "%f" del /q "%f""
else
	@rm -f $(FILES_TO_CLEAN)
endif
