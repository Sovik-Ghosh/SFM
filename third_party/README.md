# Vendored third-party libraries

Downloaded 2026-11-15. Only the files needed to build are kept (docs, samples and tests removed).

| Library | Version | Source | sha256 of download | License | Used for |
|---|---|---|---|---|---|
| stb (`stb_image.h`, `stb_image_write.h`) | commit `9600522de53973336fa033a289858a4354cbb04c` | github.com/nothings/stb | — (git clone) | MIT / public domain (`stb/LICENSE`) | PPM/PGM/PNG/JPG loading and image writing |
| nlohmann/json (`json.hpp`) | v3.11.3 | github.com/nlohmann/json releases | `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6` | MIT | JSON config |
| Eigen | 3.4.0 | gitlab.com/libeigen/eigen | `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72` (tar.gz) | MPL2 (`eigen/COPYING.*`) | small CPU linear algebra, CPU BA reference |
| Dear ImGui (core + `backends/imgui_impl_glfw.*`, `imgui_impl_opengl3*`) | v1.93.0 (downloaded 2026-11-19) | github.com/ocornut/imgui releases | `f7bf6bc76e5d79d40d751096663e6c5d18daf08129dd52e7f7b09e4251c8eeee` (tar.gz) | MIT (`imgui/LICENSE.txt`) | the local viewer `apps/sfm_viewer` (only built with `SFM_BUILD_VIEWER` and GLFW 3 + OpenGL dev packages) |
| Fonts (`fonts/`): Gelasio, IBM Plex Sans, IBM Plex Mono | google/fonts commit `06bfe0d62a98375cb09c942fa18e68bd32cff057` (downloaded 2026-11-21) | github.com/google/fonts (`ofl/gelasio`, `ofl/ibmplexsans`, `ofl/ibmplexmono`) | `Gelasio[wght].ttf` `4daecea4…68fc04e4`, `IBMPlexSans[wdth,wght].ttf` `3b031aa4…5fe2e3` (= `IBMPlexSans-Variable.ttf`, unmodified), `IBMPlexMono-Regular.ttf` `6a3412f0…7af46`, `IBMPlexMono-Medium.ttf` `a9b4c49b…6755` | SIL OFL 1.1 (`fonts/*-OFL.txt`) | the viewer's UI. `Gelasio-Regular.ttf` / `-Medium.ttf` are static instances (wght 400 / 500) of the variable file, made with fontTools 4.68 `varLib.instancer` (Gelasio has no Reserved Font Name). IBM Plex (Reserved Font Name "Plex") is shipped unmodified: the variable Plex Sans file is used at its default instance (Regular). |
| pybind11 (`include/`, `tools/`, `CMakeLists.txt`) | v3.1.0, commit `97bf890db679505a14dfe547a5e77bb2bd05dc90` (downloaded 2026-11-23) | github.com/pybind/pybind11 | — (git clone) | BSD-3 (`pybind11/LICENSE`) | the Python module `epipole` (only built with `SFM_BUILD_PYTHON`) |
| GoogleTest | v1.15.2 | github.com/google/googletest | `7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926` (tar.gz) | BSD-3 | unit tests (`ctest`) |

## Use from CMake
```cmake
set(TP ${CMAKE_SOURCE_DIR}/../third_party)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK  OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/googletest googletest)
target_include_directories(<target> SYSTEM PRIVATE ${TP}/stb ${TP}/nlohmann ${TP}/eigen)
```
Define `STB_IMAGE_IMPLEMENTATION` / `STB_IMAGE_WRITE_IMPLEMENTATION` in exactly one `.cpp` file.

Verified 2026-11-15: gcc 13.3 + nvcc 13.0.88 (sm_120), CMake 3.28; stb reads `data/bunny/images/0000.ppm`
(1024×768×3) and `silhouettes/0000.pgm` (1024×768×1); Eigen JacobiSVD, JSON parse, a CUDA kernel and gtest all pass.
