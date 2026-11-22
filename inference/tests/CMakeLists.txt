include(GoogleTest)

# sfm_add_test(<name> [SERIAL] SOURCES <files...> LIBS <targets...> LABELS <labels...>
#              [ENV <VAR=value...>])
# Labels: unit | cuda | golden | slow. Tests needing a GPU must carry "cuda".
# SERIAL: never run alongside other tests (ctest RUN_SERIAL). Use for tests that measure global
# state — free device memory, timing — which concurrent tests (ctest --parallel) would disturb.
function(sfm_add_test name)
  cmake_parse_arguments(T "SERIAL" "" "SOURCES;LIBS;LABELS;ENV" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE ${T_LIBS} GTest::gtest_main)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})  # test_env.hpp
  sfm_set_warnings(${name})
  # gtest_discover_tests (CMake 3.28) loses ';' escapes, so a label list arrives as
  # "LABELS unit cuda" and "cuda" is read as a property name. Join into one label
  # ("unit+cuda"); ctest -L / -LE take regexes, so "-LE cuda" and "-L unit" still work.
  list(JOIN T_LABELS "+" labels)
  if(T_SERIAL)
    set(serial RUN_SERIAL TRUE)
  endif()
  set(env)
  if(T_ENV)
    set(env ENVIRONMENT "${T_ENV}")
  endif()
  gtest_discover_tests(${name} PROPERTIES LABELS "${labels}" ${serial} ${env} DISCOVERY_MODE PRE_TEST)
endfunction()

add_subdirectory(core)
add_subdirectory(config)
add_subdirectory(data)
add_subdirectory(eval)
add_subdirectory(features)
add_subdirectory(matching)
add_subdirectory(recon)
add_subdirectory(ba)
add_subdirectory(pipeline)
add_subdirectory(viewer)
