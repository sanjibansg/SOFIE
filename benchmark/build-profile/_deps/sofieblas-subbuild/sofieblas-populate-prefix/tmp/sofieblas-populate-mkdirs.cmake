# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-src"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-build"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/tmp"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src"
  "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/antreas/Documents/SOFIE/benchmark/build-profile/_deps/sofieblas-subbuild/sofieblas-populate-prefix/src/sofieblas-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
