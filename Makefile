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

# Library target
LIBRARY = $(BUILD_DIR)/libcnumpy.a

# Example source files
EXAMPLE_SOURCES = $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_TARGETS = $(patsubst $(EXAMPLE_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLE_SOURCES))

# Default target
all: $(LIBRARY) examples

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/rl

# Compile library object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c cnumpy.h rl/cnumpy_rl.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

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
	rm -rf $(BUILD_DIR)

# Install (simple local install)
install: $(LIBRARY)
	mkdir -p /usr/local/lib
	mkdir -p /usr/local/include
	cp $(LIBRARY) /usr/local/lib/
	cp cnumpy.h /usr/local/include/

# Uninstall
uninstall:
	rm -f /usr/local/lib/libcnumpy.a
	rm -f /usr/local/include/cnumpy.h

# Development targets
debug: CFLAGS += -DDEBUG -g3
debug: $(LIBRARY) examples

release: CFLAGS += -O3 -DNDEBUG
release: $(LIBRARY) examples

# Static analysis
analyze:
	clang-tidy $(LIB_SOURCES) cnumpy.h

# Format code
format:
	clang-format -i $(LIB_SOURCES) cnumpy.h $(EXAMPLE_SOURCES)

# Memory check (requires valgrind)
memcheck: $(BUILD_DIR)/test_basic
	valgrind --leak-check=full --track-origins=yes $(BUILD_DIR)/test_basic

# Dependencies
$(BUILD_DIR)/cnumpy_core.o: cnumpy.h
$(BUILD_DIR)/cnumpy_ops.o: cnumpy.h
$(BUILD_DIR)/cnumpy_scope.o: cnumpy.h

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