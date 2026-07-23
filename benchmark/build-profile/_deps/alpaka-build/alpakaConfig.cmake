#
# Copyright 2021 Benjamin Worpitz, Erik Zenker, Axel Huebl, Jan Stephan, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was alpakaConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

################################################################################
# alpaka.

set(alpaka_DEBUG "0" CACHE STRING "Debug level")
set_property(CACHE alpaka_DEBUG PROPERTY STRINGS "0;1;2")

#-------------------------------------------------------------------------------
# Common.

# This file's directory.
set(_alpaka_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
get_filename_component(_alpaka_ROOT_DIR ${_alpaka_ROOT_DIR} ABSOLUTE)

# Compiler feature tests.
set(_alpaka_FEATURE_TESTS_DIR "${_alpaka_ROOT_DIR}/tests")

# Add common functions.
set(_alpaka_COMMON_FILE "${_alpaka_ROOT_DIR}/common.cmake")
include(${_alpaka_COMMON_FILE})

# Add alpaka_ADD_EXECUTABLE function.
set(_alpaka_ADD_EXECUTABLE_FILE "${_alpaka_ROOT_DIR}/addExecutable.cmake")
include(${_alpaka_ADD_EXECUTABLE_FILE})

# Add alpaka_ADD_LIBRARY function.
set(_alpaka_ADD_LIBRARY_FILE "${_alpaka_ROOT_DIR}/addLibrary.cmake")
include(${_alpaka_ADD_LIBRARY_FILE})

# Set found to true initially and set it to false if a required dependency is missing.
set(_alpaka_FOUND TRUE)

set(_alpaka_INCLUDE_DIRECTORY "/usr/local/include")

include("${CMAKE_CURRENT_LIST_DIR}/alpakaCommon.cmake")

check_required_components("alpaka")

# Unset already set variables if not found.
if(NOT _alpaka_FOUND)
    unset(_alpaka_FOUND)
    unset(_alpaka_COMPILE_OPTIONS_PUBLIC)
    unset(_alpaka_COMPILE_DEFINITIONS_HIP)
    unset(_alpaka_HIP_LIBRARIES)
    unset(_alpaka_INCLUDE_DIRECTORY)
    unset(_alpaka_ADD_EXECUTABLE_FILE)
    unset(_alpaka_ADD_LIBRARY_FILE)
    unset(_alpaka_BOOST_MIN_VER)
else()
    # Make internal variables advanced options in the GUI.
    mark_as_advanced(
        _alpaka_COMPILE_OPTIONS_PUBLIC
        _alpaka_INCLUDE_DIRECTORY
        _alpaka_COMMON_FILE
        _alpaka_ADD_EXECUTABLE_FILE
        _alpaka_ADD_LIBRARY_FILE
        _alpaka_BOOST_MIN_VER)
endif()
