cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_wrapper)

# Add custom install directory.
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../LibLocal" CACHE PATH "Default install path" FORCE )
endif()

# The library prefix
set(LIB_PREFIX torch_wrapper_)

# Default built type
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Postfix, based on type
set(CMAKE_DEBUG_POSTFIX "_d" CACHE STRING "postfix applied to debug build of libraries")
set(CMAKE_RELEASE_POSTFIX "" CACHE STRING "postfix applied to release build of libraries")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd" CACHE STRING "postfix applied to release-with-debug-information libraries")
set(CMAKE_MINSIZEREL_POSTFIX "_s" CACHE STRING "postfix applied to minimium-size-build libraries")

# work out the postfix; required where we use OUTPUT_NAME
IF(CMAKE_BUILD_TYPE MATCHES Release)
  set(EXE_POSTFIX)
ELSEIF(CMAKE_BUILD_TYPE MATCHES Debug)
  set(EXE_POSTFIX ${CMAKE_DEBUG_POSTFIX})
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(EXE_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
ELSEIF(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(EXE_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})
ENDIF(CMAKE_BUILD_TYPE MATCHES Release)

# Allow the developer to select if Dynamic or Static libraries are built
OPTION (BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
set (TORCH_WRAPPER_LIB_TYPE STATIC)
IF (BUILD_SHARED_LIBS)
  set (TORCH_WRAPPER_LIB_TYPE SHARED)
ENDIF()

# Set the output directory for the build executables and libraries
set(torch_wrapper_RUNTIME_OUTPUT_DIRECTORY ${torch_wrapper_SOURCE_DIR}/bin CACHE PATH "Target for the binaries")
IF(WIN32)
  set(torch_wrapper_LIBRARY_OUTPUT_DIRECTORY ${torch_wrapper_SOURCE_DIR}/bin CACHE PATH "Target for the libraries")
ELSE(WIN32)
  set(torch_wrapper_LIBRARY_OUTPUT_DIRECTORY ${torch_wrapper_SOURCE_DIR}/lib CACHE PATH "Target for the libraries")
ENDIF(WIN32)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${torch_wrapper_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${torch_wrapper_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${torch_wrapper_RUNTIME_OUTPUT_DIRECTORY})

# Set standard installation directories
set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/torch_wrapper)

# Set search directory for looking for our custom CMake scripts to
# look for grpc, protobuf
#LIST(APPEND CMAKE_MODULE_PATH ${corres_SOURCE_DIR}/cmake)

# Detect OS and define macros appropriately
IF(WIN32)
  ADD_DEFINITIONS(-DWINDOWS)
  MESSAGE(STATUS "Compiling on Windows")
ELSEIF(CYGWIN)
  MESSAGE(STATUS "Compiling on Cygwin")
  ADD_DEFINITIONS(-DCYGWIN)
ELSEIF(APPLE)
  ADD_DEFINITIONS(-DUNIX)
  MESSAGE(STATUS "Compiling on OSX")
ELSEIF(UNIX)
  ADD_DEFINITIONS(-DUNIX)
  MESSAGE(STATUS "Compiling on Unix")
ENDIF(WIN32)

# C++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Specifying compiler flags.
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add global preprocessor definitions.
add_definitions(-DPROJECT_DIR="${torch_wrapper_SOURCE_DIR}")
add_definitions(-DENABLE_ASSERTIONS)
add_definitions(-DCOMPILE_CUDA)
add_definitions(-DCUDA_ERROR_CHECK)

# Set up the top-level include directories.
# We add before to make prioritize local files to globally (already installed) library headers.
include_directories(BEFORE ${torch_wrapper_SOURCE_DIR})

# Include local external dependencies.
include_directories(../LibLocal/include)

# Include library dependencies.
find_package(Torch REQUIRED)

# Setup a static library.
set(TW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/torch_wrapper")
file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/torch_wrapper/*.*" "${CMAKE_CURRENT_SOURCE_DIR}/torch_wrapper/internal/*.*")            
source_group(TREE ${TW_SRC} PREFIX "Source Files" FILES ${sources} )

add_library(torch_wrapper STATIC ${sources})
target_link_libraries(torch_wrapper ${TORCH_LIBRARIES})

install(TARGETS torch_wrapper
  EXPORT torch_wrapper
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
  INCLUDES DESTINATION ${INCLUDES_DESTINATION}
)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/torch_wrapper/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/torch_wrapper/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/torch_wrapper/*.cuh")

install(FILES ${headers} DESTINATION ${INCLUDES_INSTALL_DIR})

# Setup the executable.
add_executable(torch_wrapper_main main.cpp)
target_link_libraries(torch_wrapper_main ${TORCH_LIBRARIES})
set_property(TARGET torch_wrapper_main PROPERTY CXX_STANDARD 11)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch_wrapper_main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch_wrapper_main>)
endif (MSVC)