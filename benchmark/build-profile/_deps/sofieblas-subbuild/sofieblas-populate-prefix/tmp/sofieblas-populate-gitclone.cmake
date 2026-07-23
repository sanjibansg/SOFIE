# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitclone-lastrun.txt" AND EXISTS "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitinfo.txt" AND
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/ML4EP/sofieBLAS" "sofieblas-src"
    WORKING_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/ML4EP/sofieBLAS'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "dev" --
  WORKING_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'dev'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitinfo.txt" "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/sofieblas-populate-gitclone-lastrun.txt'")
endif()
