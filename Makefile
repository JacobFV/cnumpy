# CNmpy Makefile
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -g
LDFLAGS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build
EXAMPLE_DIR = examples

# Library source files
LIB_SOURCES = cnumpy_core.c cnumpy_ops.c cnumpy_scope.c rl/cnumpy_rl_core.c rl/cnumpy_rl_env.c rl/cnumpy_rl_agents.c
LIB_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SOURCES))

# Automatically detect subdirectories from source files
LIB_SUBDIRS = $(sort $(filter-out ./,$(dir $(LIB_SOURCES))))
BUILD_SUBDIRS = $(addprefix $(BUILD_DIR)/,$(patsubst %/,%,$(LIB_SUBDIRS)))

# Library target
LIBRARY = $(BUILD_DIR)/libcnumpy.a

# Example source files
EXAMPLE_SOURCES = $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_TARGETS = $(patsubst $(EXAMPLE_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLE_SOURCES))

# Default target
all: $(LIBRARY) examples

# Create build directory and all necessary subdirectories
$(BUILD_DIR):
ifeq ($(OS),Windows_NT)
	powershell -Command "New-Item -Path '$(BUILD_DIR)' -ItemType Directory -Force"
	$(if $(BUILD_SUBDIRS), powershell -Command "$(foreach dir,$(BUILD_SUBDIRS),New-Item -Path '$(dir)' -ItemType Directory -Force; )")
else
	mkdir -p $(BUILD_DIR)
	$(if $(BUILD_SUBDIRS), mkdir -p $(BUILD_SUBDIRS))
endif

# Compile library object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/cnumpy.h $(SRC_DIR)/rl/cnumpy_rl.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Create static library
$(LIBRARY): $(LIB_OBJECTS)
	ar rcs $@ $^

# Compile examples
examples: $(EXAMPLE_TARGETS)

$(BUILD_DIR)/%: $(EXAMPLE_DIR)/%.c $(LIBRARY)
	$(CC) $(CFLAGS) $< -L$(BUILD_DIR) -lcnumpy $(LDFLAGS) -o $@

# Test targets
test: $(BUILD_DIR)/test_basic
	$(BUILD_DIR)/test_basic

# Clean build artifacts
clean:
ifeq ($(OS),Windows_NT)
	powershell -Command "if (Test-Path '$(BUILD_DIR)') { Remove-Item '$(BUILD_DIR)' -Recurse -Force }"
else
	rm -rf $(BUILD_DIR)
endif

# Install (simple local install)
install: $(LIBRARY)
	mkdir -p /usr/local/lib
	mkdir -p /usr/local/include
	cp $(LIBRARY) /usr/local/lib/
	cp $(SRC_DIR)/cnumpy.h /usr/local/include/

# Uninstall
uninstall:
ifeq ($(OS),Windows_NT)
	powershell -Command "if (Test-Path '/usr/local/lib/libcnumpy.a') { Remove-Item '/usr/local/lib/libcnumpy.a' -Force }"
	powershell -Command "if (Test-Path '/usr/local/include/cnumpy.h') { Remove-Item '/usr/local/include/cnumpy.h' -Force }"
else
	rm -f /usr/local/lib/libcnumpy.a
	rm -f /usr/local/include/cnumpy.h
endif

# Development targets
debug: CFLAGS += -DDEBUG -g3
debug: $(LIBRARY) examples

release: CFLAGS += -O3 -DNDEBUG
release: $(LIBRARY) examples

# Static analysis
analyze:
	clang-tidy $(addprefix $(SRC_DIR)/,$(LIB_SOURCES)) $(SRC_DIR)/cnumpy.h

# Format code
format:
	clang-format -i $(addprefix $(SRC_DIR)/,$(LIB_SOURCES)) $(SRC_DIR)/cnumpy.h $(EXAMPLE_SOURCES)

# Memory check (requires valgrind)
memcheck: $(BUILD_DIR)/test_basic
	valgrind --leak-check=full --track-origins=yes $(BUILD_DIR)/test_basic

# Dependencies
$(BUILD_DIR)/cnumpy_core.o: $(SRC_DIR)/cnumpy.h
$(BUILD_DIR)/cnumpy_ops.o: $(SRC_DIR)/cnumpy.h
$(BUILD_DIR)/cnumpy_scope.o: $(SRC_DIR)/cnumpy.h

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build library and examples"
	@echo "  $(LIBRARY)  - Build static library"
	@echo "  examples    - Build all examples"
	@echo "  test        - Run basic tests"
	@echo "  clean       - Remove build artifacts"
	@echo "  install     - Install library and headers"
	@echo "  uninstall   - Remove installed files"
	@echo "  debug       - Build with debug flags"
	@echo "  release     - Build with optimizations"
	@echo "  analyze     - Run static analysis"
	@echo "  format      - Format source code"
	@echo "  memcheck    - Run memory leak check"
	@echo "  help        - Show this help message"

.PHONY: all examples test clean install uninstall debug release analyze format memcheck help 