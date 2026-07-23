# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-src"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-build"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/tmp"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/alpaka-subbuild/alpaka-populate-prefix/src/alpaka-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
