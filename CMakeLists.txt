cmake_minimum_required(VERSION 3.12)

# Project configuration
project(CNmpy_RL
    VERSION 1.0.0
    DESCRIPTION "High-Performance Reinforcement Learning Library for C"
    LANGUAGES C
)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Build options
option(ENABLE_TESTING "Enable testing" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(ENABLE_DOCUMENTATION "Enable documentation generation" OFF)
option(ENABLE_EXAMPLES "Build examples" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_SIMD "Enable SIMD optimizations" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)

# Optional packages
if(ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

if(ENABLE_DOCUMENTATION)
    find_package(Doxygen QUIET)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g3 -O0 -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# Warning flags
set(CNUMPY_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wcast-align
    -Wcast-qual
    -Wformat=2
    -Wformat-security
    -Wmissing-declarations
    -Wmissing-prototypes
    -Wnull-dereference
    -Wpointer-arith
    -Wredundant-decls
    -Wshadow
    -Wstrict-prototypes
    -Wunused
    -Wwrite-strings
)

# Performance flags
set(CNUMPY_PERFORMANCE_FLAGS)
if(ENABLE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND CNUMPY_PERFORMANCE_FLAGS -march=native -mtune=native)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        list(APPEND CNUMPY_PERFORMANCE_FLAGS -march=native -mtune=native)
    endif()
endif()

# Sanitizer flags
set(CNUMPY_SANITIZER_FLAGS)
if(ENABLE_SANITIZERS)
    list(APPEND CNUMPY_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize-address-use-after-scope
        -fno-omit-frame-pointer
    )
endif()

# Coverage flags
set(CNUMPY_COVERAGE_FLAGS)
if(ENABLE_COVERAGE)
    list(APPEND CNUMPY_COVERAGE_FLAGS
        --coverage
        -fprofile-arcs
        -ftest-coverage
    )
endif()

# Source files - automatically discover all .c files
file(GLOB CNUMPY_CORE_SOURCES
    "src/*.c"
)

file(GLOB CNUMPY_RL_SOURCES
    "src/rl/*.c"
)

set(CNUMPY_ALL_SOURCES ${CNUMPY_CORE_SOURCES} ${CNUMPY_RL_SOURCES})

# Header files - automatically discover all .h files
file(GLOB CNUMPY_HEADERS
    "src/*.h"
    "src/rl/*.h"
)

# Create main library
add_library(cnumpy ${CNUMPY_ALL_SOURCES})

# Set library properties
set_target_properties(cnumpy PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${CNUMPY_HEADERS}"
    OUTPUT_NAME "cnumpy"
)

# Include directories
target_include_directories(cnumpy
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rl
)

# Link libraries
target_link_libraries(cnumpy
    PUBLIC
        Threads::Threads
        m  # Math library
)

# OpenMP support
if(ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(cnumpy PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(cnumpy PRIVATE CNUMPY_OPENMP_ENABLED)
endif()

# Apply compiler flags
target_compile_options(cnumpy PRIVATE
    ${CNUMPY_WARNING_FLAGS}
    ${CNUMPY_PERFORMANCE_FLAGS}
    ${CNUMPY_SANITIZER_FLAGS}
    ${CNUMPY_COVERAGE_FLAGS}
)

# Apply linker flags
target_link_options(cnumpy PRIVATE
    ${CNUMPY_SANITIZER_FLAGS}
    ${CNUMPY_COVERAGE_FLAGS}
)

# Create alias for consistency
add_library(CNmpy::cnumpy ALIAS cnumpy)

# Examples
if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

# Testing
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Documentation
if(ENABLE_DOCUMENTATION AND DOXYGEN_FOUND)
    add_subdirectory(docs)
endif()

# Installation
include(GNUInstallDirs)

# Install library
install(TARGETS cnumpy
    EXPORT CNmpyTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cnumpy
)

# Install headers
install(FILES src/cnumpy.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cnumpy
)

install(FILES src/rl/cnumpy_rl.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cnumpy/rl
)

# Install examples
if(ENABLE_EXAMPLES)
    install(DIRECTORY examples/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/cnumpy/examples
        FILES_MATCHING PATTERN "*.c"
    )
endif()

# Install documentation
install(FILES README.md CONTRIBUTING.md LICENSE CHANGELOG.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

# Create and install package configuration files
include(CMakePackageConfigHelpers)

# Generate package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CNmpyConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CNmpyConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CNmpy
)

# Generate package version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CNmpyConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install package config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CNmpyConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CNmpyConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CNmpy
)

# Install export targets
install(EXPORT CNmpyTargets
    FILE CNmpyTargets.cmake
    NAMESPACE CNmpy::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CNmpy
)

# Create uninstall target
if(NOT TARGET uninstall)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        IMMEDIATE @ONLY
    )
    
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

# Package generation
set(CPACK_PACKAGE_NAME "CNmpy-RL")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "CNmpy RL Team")
set(CPACK_PACKAGE_CONTACT "support@cnumpy-rl.org")
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)

# Platform-specific package configuration
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "CNmpy RL")
    set(CPACK_NSIS_PACKAGE_NAME "CNmpy-RL")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/username/cnumpy-rl")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "CNmpy-RL")
    set(CPACK_DMG_FORMAT "UDBZ")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    
    # Debian package configuration
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.17), libgcc1 (>= 1:4.1.1)")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/username/cnumpy-rl")
    
    # RPM package configuration
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/username/cnumpy-rl")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.17, libgcc >= 4.1.1")
endif()

include(CPack)

# Development tools
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Add clang-format target
    find_program(CLANG_FORMAT clang-format)
    if(CLANG_FORMAT)
        file(GLOB_RECURSE ALL_SOURCE_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        )
        
        add_custom_target(format
            COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCE_FILES}
            COMMENT "Running clang-format"
        )
        
        add_custom_target(format-check
            COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ALL_SOURCE_FILES}
            COMMENT "Checking code formatting"
        )
    endif()
    
    # Add clang-tidy target
    find_program(CLANG_TIDY clang-tidy)
    if(CLANG_TIDY)
        add_custom_target(tidy
            COMMAND ${CLANG_TIDY} ${CNUMPY_ALL_SOURCES} -- -I${CMAKE_CURRENT_SOURCE_DIR}/src
            COMMENT "Running clang-tidy"
        )
    endif()
    
    # Add valgrind target
    find_program(VALGRIND valgrind)
    if(VALGRIND AND ENABLE_TESTING)
        add_custom_target(memcheck
            COMMAND ${VALGRIND} --leak-check=full --track-origins=yes --error-exitcode=1 
                    ${CMAKE_CURRENT_BINARY_DIR}/tests/test_basic
            DEPENDS tests
            COMMENT "Running valgrind memory check"
        )
    endif()
endif()

# Coverage reporting
if(ENABLE_COVERAGE)
    find_program(GCOV gcov)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)
    
    if(GCOV AND LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${LCOV} --list coverage.info
            COMMAND ${GENHTML} coverage.info --output-directory coverage_html
            COMMENT "Generating coverage report"
        )
    endif()
endif()

# Summary
message(STATUS "CNmpy RL Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Testing: ${ENABLE_TESTING}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "  Documentation: ${ENABLE_DOCUMENTATION}")
message(STATUS "  Examples: ${ENABLE_EXAMPLES}")
message(STATUS "  Benchmarks: ${ENABLE_BENCHMARKS}")
message(STATUS "  OpenMP: ${ENABLE_OPENMP}")
message(STATUS "  SIMD: ${ENABLE_SIMD}")
message(STATUS "  Shared libraries: ${BUILD_SHARED_LIBS}") 