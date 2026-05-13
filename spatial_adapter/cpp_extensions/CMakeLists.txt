cmake_minimum_required(VERSION 3.18)
project(spatial_utils)

# === C++ Config ===
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Armadillo: Disable MKL fallback ===
add_definitions(-DARMA_DONT_USE_WRAPPER)
set(ENV{ARMA_DONT_USE_WRAPPER} 1)

# === Python & Pybind11 ===
# scikit-build-core injects Python + pybind11 paths when building wheels.
# find_package(Python) works with 3.10/3.11/3.12 transparently so cibuildwheel
# can build across versions without pinning to 3.10.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# === Armadillo & BLAS/LAPACK ===
# 1) macOS Homebrew (arm64 /opt/homebrew or Intel /usr/local)
# 2) Conda (CONDA_PREFIX set)
# 3) Linux system libs (e.g. apt: libarmadillo-dev libopenblas-dev)
if(APPLE)
    # Resolve Homebrew prefix dynamically (arm64 vs Intel)
    execute_process(COMMAND brew --prefix
                    OUTPUT_VARIABLE HOMEBREW_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT HOMEBREW_PREFIX)
        set(HOMEBREW_PREFIX "/opt/homebrew")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")

    # armadillo — may or may not be keg-only depending on version
    find_library(ARMADILLO_LIBRARIES NAMES armadillo
        PATHS "${HOMEBREW_PREFIX}/lib"
              "${HOMEBREW_PREFIX}/opt/armadillo/lib"
        NO_DEFAULT_PATH)
    find_path(ARMADILLO_INCLUDE_DIRS armadillo
        PATHS "${HOMEBREW_PREFIX}/include"
              "${HOMEBREW_PREFIX}/opt/armadillo/include"
        NO_DEFAULT_PATH)

    # openblas is keg-only on macOS — not linked into HOMEBREW_PREFIX/lib
    find_library(OPENBLAS_LIB NAMES openblas
        PATHS "${HOMEBREW_PREFIX}/lib"
              "${HOMEBREW_PREFIX}/opt/openblas/lib"
        NO_DEFAULT_PATH)

    if(NOT ARMADILLO_LIBRARIES)
        message(FATAL_ERROR "armadillo not found. Run: brew install armadillo")
    endif()
    if(NOT OPENBLAS_LIB)
        message(FATAL_ERROR "openblas not found. Run: brew install openblas")
    endif()
    set(LAPACK_LIB "${OPENBLAS_LIB}")
elseif(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(ARMADILLO_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    if(APPLE)
        set(ARMADILLO_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libarmadillo.dylib")
        set(OPENBLAS_LIB "$ENV{CONDA_PREFIX}/lib/libopenblas.dylib")
    else()
        set(ARMADILLO_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libarmadillo.so")
        set(OPENBLAS_LIB "$ENV{CONDA_PREFIX}/lib/libopenblas.so")
    endif()
    find_library(LAPACK_LIB NAMES lapack PATHS "$ENV{CONDA_PREFIX}/lib" NO_DEFAULT_PATH)
    if(NOT LAPACK_LIB)
        set(LAPACK_LIB ${OPENBLAS_LIB})
    endif()
else()
    # Linux system packages (e.g. apt install libarmadillo-dev libopenblas-dev)
    find_package(Armadillo REQUIRED)
    find_library(OPENBLAS_LIB NAMES openblas)
    if(NOT OPENBLAS_LIB)
        message(FATAL_ERROR "OpenBLAS not found. Install libopenblas-dev.")
    endif()
    find_library(LAPACK_LIB NAMES lapack)
    if(NOT LAPACK_LIB)
        set(LAPACK_LIB ${OPENBLAS_LIB})
    endif()
    # ARMADILLO_INCLUDE_DIRS, ARMADILLO_LIBRARIES set by FindArmadillo
endif()

# === Module Build ===
pybind11_add_module(spatial_utils spatial_utils.cpp)

# === Compiler/Linker Flags ===
target_include_directories(spatial_utils PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_compile_definitions(spatial_utils PRIVATE ARMA_DONT_USE_WRAPPER)
target_link_libraries(spatial_utils PRIVATE
    ${ARMADILLO_LIBRARIES}
    ${OPENBLAS_LIB}
    ${LAPACK_LIB}
)

# === Output Location ===
# Build output lands in the source tree so `make build-cpp` + editable
# installs pick it up at spatial_adapter/cpp_extensions/spatial_utils.so.
set_target_properties(spatial_utils PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_NAME "spatial_utils"
    SUFFIX ".so"
)

# === Install (for scikit-build-core wheels) ===
# When building a wheel, install() copies the compiled module into the
# right spot inside the wheel so `pip install spatial-adapter` ships the
# .so without any post-install build step.
install(TARGETS spatial_utils
        LIBRARY DESTINATION spatial_adapter/cpp_extensions)
