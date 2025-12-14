add_library(utils INTERFACE)

target_include_directories(utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SOFIE
    DESTINATION include
)
