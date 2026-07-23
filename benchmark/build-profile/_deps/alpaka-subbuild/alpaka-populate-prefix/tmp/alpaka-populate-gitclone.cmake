# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitclone-lastrun.txt" AND EXISTS "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitinfo.txt" AND
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/alpaka-group/alpaka" "alpaka-src"
    WORKING_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/alpaka-group/alpaka'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "2fa91a34ed11b2076e474c5507d920e85cf9b79d" --
  WORKING_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '2fa91a34ed11b2076e474c5507d920e85cf9b79d'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitinfo.txt" "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/alpaka-populate-gitclone-lastrun.txt'")
endif()
