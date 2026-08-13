# Python / pybind11 bindings subdirectory
# Builds the C++ extension module (turbol_core) and configures the Python package.

cmake_minimum_required(VERSION 3.18)

# --------------------------------------------------------------------------- #
# Find pybind11
# --------------------------------------------------------------------------- #
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "pybind11 include:  ${pybind11_INCLUDE_DIR}")

# pybind11's pybind11_add_module() relies on Python_add_library, provided by
# FindPython (Development.Module). Without this, configuration fails with
# "Unknown CMake command python_add_library".
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

# --------------------------------------------------------------------------- #
# Compile the C++ extension
# --------------------------------------------------------------------------- #
pybind11_add_module(turbol_core
    turbol/pybind11_bindings.cpp
    # Include the same C++ sources as the main library (but only headers
    # are needed for the bindings — no source files needed here since
    # turbol::Tensor etc. are header-only or already compiled into libturbol.a)
)

# Link against the compiled turbol static library
target_link_libraries(turbol_core PRIVATE turbol)

# pybind11 + libtorch support
if(TURBORL_USE_LIBTORCH)
    target_link_libraries(turbol_core PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(turbol_core PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

# CUDA support for bindings
if(TURBORL_ENABLE_CUDA)
    target_link_libraries(turbol_core PRIVATE CUDA::cudart)
    target_compile_options(turbol_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr>)
endif()

# Standard and warnings
target_compile_features(turbol_core PRIVATE cxx_std_17)
target_include_directories(turbol_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/python
)

# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
# Install the compiled .so / .pyd next to the Python sources
install(TARGETS turbol_core
    LIBRARY DESTINATION turbol
    ARCHIVE DESTINATION turbol
)

# Install Python source files
install(FILES
    turbol/__init__.py
    turbol/py.typed
    DESTINATION turbol
)

# Mark as a pip-installable package
install(CODE [[
    message(STATUS "TurboRL Python package installed.")
    message(STATUS "  import turbol  # C++ core: turbol_core.*")
]])

# --------------------------------------------------------------------------- #
# Development helpers
# --------------------------------------------------------------------------- #
# "pip install -e ." support
add_custom_target(install_python_editable
    COMMAND ${Python3_EXECUTABLE} -m pip install --no-build-isolation -e .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing turbol in editable mode"
)

message(STATUS "Python bindings: turbol_core target created")
message(STATUS "  Build in-place:  ninja turbol_core")
message(STATUS "  Install editable: ninja install_python_editable")
message(STATUS "  Or:              pip install -e python/")
