# Fallback test macros used when ROOT is not available.
# These provide the same interface as ROOTTEST_GENERATE_EXECUTABLE and
# ROOTTEST_ADD_TEST from RoottestMacros.cmake but without requiring ROOT.

macro(ROOTTEST_GENERATE_EXECUTABLE executable)
  cmake_parse_arguments(ARG "" "RESOURCE_LOCK"
    "LIBRARIES;COMPILE_FLAGS;DEPENDS;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED"
    ${ARGN})

  add_executable(${executable} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  set_target_properties(${executable} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(ARG_DEPENDS)
    add_dependencies(${executable} ${ARG_DEPENDS})
  endif()

  if(ARG_LIBRARIES)
    target_link_libraries(${executable} ${ARG_LIBRARIES})
  endif()

  if(ARG_COMPILE_FLAGS)
    set_target_properties(${executable} PROPERTIES COMPILE_FLAGS ${ARG_COMPILE_FLAGS})
  endif()

  set(_sofie_build_test ${executable}-build)
  add_test(NAME ${_sofie_build_test}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${executable})

  if(ARG_FIXTURES_SETUP)
    set_property(TEST ${_sofie_build_test} PROPERTY FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()
  if(ARG_FIXTURES_CLEANUP)
    set_property(TEST ${_sofie_build_test} PROPERTY FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()
  if(ARG_FIXTURES_REQUIRED)
    set_property(TEST ${_sofie_build_test} PROPERTY FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()
endmacro()

function(ROOTTEST_ADD_TEST testname)
  cmake_parse_arguments(ARG ""
    "WORKING_DIR;TIMEOUT;RESOURCE_LOCK"
    "EXEC;COMMAND;DEPENDS;FIXTURES_SETUP;FIXTURES_CLEANUP;FIXTURES_REQUIRED;ENVIRONMENT;PROPERTIES"
    ${ARGN})

  if(ARG_EXEC)
    set(_cmd ${ARG_EXEC})
  elseif(ARG_COMMAND)
    set(_cmd ${ARG_COMMAND})
  else()
    message(FATAL_ERROR "ROOTTEST_ADD_TEST: must specify EXEC or COMMAND")
  endif()

  add_test(NAME ${testname} COMMAND ${_cmd}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(ARG_FIXTURES_SETUP)
    set_property(TEST ${testname} PROPERTY FIXTURES_SETUP ${ARG_FIXTURES_SETUP})
  endif()
  if(ARG_FIXTURES_CLEANUP)
    set_property(TEST ${testname} PROPERTY FIXTURES_CLEANUP ${ARG_FIXTURES_CLEANUP})
  endif()
  if(ARG_FIXTURES_REQUIRED)
    set_property(TEST ${testname} PROPERTY FIXTURES_REQUIRED ${ARG_FIXTURES_REQUIRED})
  endif()
  if(ARG_ENVIRONMENT)
    set_property(TEST ${testname} PROPERTY ENVIRONMENT ${ARG_ENVIRONMENT})
  endif()
  if(ARG_TIMEOUT)
    set_property(TEST ${testname} PROPERTY TIMEOUT ${ARG_TIMEOUT})
  endif()
endfunction()
