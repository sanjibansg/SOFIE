cmake_minimum_required(VERSION 3.16)
project(Sofie
    VERSION 1.0.1
    DESCRIPTION "SOFIE"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(BLAS)
if(NOT BLAS_FOUND)
  message(WARNING "BLAS not found: sofie will not be fully tested")
endif()

message(STATUS "Looking for Protobuf")
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG)
if(NOT Protobuf_FOUND)
  find_package(Protobuf MODULE)
endif()
if(NOT Protobuf_FOUND)
  if(fail-on-missing)
    message(FATAL_ERROR "Protobuf libraries not found and they are required (sofie option enabled)")
  else()
    message(STATUS "Protobuf not found. Switching off sofie option")
    message(FATAL_ERROR "SOFIE cannot be installed without Protobuf")
  endif()
else()
  if(Protobuf_VERSION LESS 3.0)
    if(fail-on-missing)
      message(FATAL_ERROR "Protobuf libraries found but is less than the version required (3.0) (sofie option enabled)")
    else()
      message(STATUS "Protobuf found but its version is not high enough (>3.0). Switching off sofie option")
      message(FATAL_ERROR "SOFIE cannot be installed without Protobuf")
    endif()
  else()
    if(NOT TARGET protobuf::protoc)
      if(fail-on-missing)
        message(FATAL_ERROR "Protobuf compiler not found (sofie option enabled)")
      else()
        message(STATUS "Protobuf compiler not found. Switching off sofie option")
        message(FATAL_ERROR "SOFIE cannot be installed without Protobuf")
      endif()
    endif()
  endif()
endif()

option(SOFIE_WITH_ROOT "Enable ROOT support (required for .root weight files and ROOT serialization)" OFF)

if(SOFIE_WITH_ROOT)
  find_package(ROOT REQUIRED COMPONENTS Core TMVA Tree)
  if(ROOT_FOUND)
    include(${ROOT_USE_FILE})
    message(STATUS "ROOT found: enabling ROOT support in SOFIE")
  else()
    message(FATAL_ERROR "SOFIE_WITH_ROOT is ON but ROOT was not found")
  endif()
else()
  message(STATUS "Building SOFIE without ROOT support (SOFIE_WITH_ROOT=OFF)")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

option(testing "Build and run tests" OFF)
if(testing)
  find_package(GTest REQUIRED)
  enable_testing()
endif()

option(SOFIE_BENCHMARK "Build the SOFIE CUDA benchmark toolkit" OFF)

if(SOFIE_WITH_ROOT AND ROOT_FOUND)
  include(cmake/modules/RoottestMacros.cmake)
else()
  include(cmake/modules/SofieTestMacros.cmake)
endif()

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(parsers)

if(SOFIE_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# ── Install cmake package config files ──────────────────────────────────────

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SOFIEConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SOFIE
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(
  EXPORT SOFIETargets
  FILE SOFIETargets.cmake
  NAMESPACE SOFIE::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SOFIE
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SOFIE
)
