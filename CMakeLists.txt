cmake_minimum_required(VERSION 3.16)
project(morphserve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

add_compile_options(-O2 -Wall -Wextra -Wpedantic)

set(MORPHSERVE_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/deploy/kairos/deps/nlohmann3.11.3"
)

add_library(morphserve_core SHARED
    src/morphserve/runtime/cuda_driver.cpp
    src/morphserve/runtime/numa_mirror.cpp
    src/morphserve/runtime/h2d_pacer.cpp
    src/morphserve/runtime/peer_transport.cpp
    src/morphserve/vmm/page_backend.cpp
    src/morphserve/vmm/tripool_vmm.cpp
    src/morphserve/control/hash_util.cpp
    src/morphserve/control/topology.cpp
    src/morphserve/control/policy.cpp
    src/morphserve/control/budget_planner.cpp
    src/morphserve/control/gpu_queue_scheduler.cpp
    src/morphserve/control/performance_atlas.cpp
    src/morphserve/control/preempt_edf.cpp
    src/morphserve/control/router.cpp
    src/morphserve/control/morph_controller.cpp
    src/morphserve/control/sim_memory_ledger.cpp
    src/morphserve/control/sim_trace_loader.cpp
    src/morphserve/control/sim_calibration.cpp
    src/morphserve/control/simulator.cpp
    src/morphserve/c_api.cpp
)
target_include_directories(morphserve_core PUBLIC ${MORPHSERVE_INCLUDE_DIRS})
target_link_libraries(morphserve_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

add_custom_command(TARGET morphserve_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_NAME:morphserve_core>
        "$<TARGET_FILE_DIR:morphserve_core>/libtripool_vmm.so"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_NAME:morphserve_core>
        "$<TARGET_FILE_DIR:morphserve_core>/libtripool_pluggable.so"
)

add_executable(morphserve-sim cmd/morphserve_sim.cpp)
target_include_directories(morphserve-sim PRIVATE ${MORPHSERVE_INCLUDE_DIRS})
target_link_libraries(morphserve-sim PRIVATE morphserve_core)
set_target_properties(morphserve-sim PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

add_executable(tripool_vmm_probe cmd/tripool_vmm_probe.cpp)
target_include_directories(tripool_vmm_probe PRIVATE ${MORPHSERVE_INCLUDE_DIRS})
target_link_libraries(tripool_vmm_probe PRIVATE morphserve_core ${CMAKE_DL_LIBS})
set_target_properties(tripool_vmm_probe PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

install(TARGETS morphserve_core
    LIBRARY DESTINATION lib
)
install(TARGETS morphserve-sim tripool_vmm_probe
    RUNTIME DESTINATION bin
)
install(CODE "
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libmorphserve_core.so \"\${CMAKE_INSTALL_PREFIX}/lib/libtripool_vmm.so\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libmorphserve_core.so \"\${CMAKE_INSTALL_PREFIX}/lib/libtripool_pluggable.so\")
")

option(MORPHSERVE_BUILD_TESTS "Build MorphServe native C++17 CTest suites" ON)
if(MORPHSERVE_BUILD_TESTS)
    enable_testing()

    foreach(test_name
        test_tripool_vmm
        test_runtime_transport
        test_control_engine
        test_cluster_simulator
        test_c_api
    )
        add_executable(${test_name} "tests/cpp/${test_name}.cpp")
        target_include_directories(${test_name} PRIVATE ${MORPHSERVE_INCLUDE_DIRS})
        target_link_libraries(${test_name} PRIVATE morphserve_core)
        add_test(NAME ${test_name} COMMAND ${test_name}
                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endforeach()
endif()
