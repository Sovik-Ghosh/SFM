# Python module `epipole` (pybind11): a thin layer over sfm::pipeline. Built with
# -DSFM_BUILD_PYTHON=ON; `pip install .` (pyproject.toml, scikit-build-core) does that and installs
# only the component "python".
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11
                 EXCLUDE_FROM_ALL)

# The tuned bunny config is the template of epipole.reconstruct().
file(READ ${PROJECT_SOURCE_DIR}/configs/bunny_sift_tuned2.json SFM_DEFAULT_CONFIG)
configure_file(default_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/default_config.hpp @ONLY)

pybind11_add_module(_epipole src/module.cpp)
target_link_libraries(_epipole PRIVATE sfm::pipeline sfm::json)
target_include_directories(_epipole PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
sfm_set_warnings(_epipole)
install(TARGETS _epipole LIBRARY DESTINATION epipole COMPONENT python)

# Tests: the package (python/epipole + the built module) staged next to each other in the build tree.
if(SFM_BUILD_TESTS)
  set(stage ${CMAKE_CURRENT_BINARY_DIR}/stage)
  add_custom_command(TARGET _epipole POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${stage}/epipole
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/epipole/__init__.py ${stage}/epipole/
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_epipole> ${stage}/epipole/)
  add_test(NAME python.epipole
    COMMAND Python::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_epipole.py ${PROJECT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/test_work)
  set_tests_properties(python.epipole PROPERTIES ENVIRONMENT "PYTHONPATH=${stage}" LABELS "unit")
endif()
