# sfm_set_warnings(<target>)
# Strict warnings for project code only (never applied to third_party).
function(sfm_set_warnings target)
  set(cxx_flags -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast)
  set(cuda_flags -Xcompiler=-Wall,-Wextra)
  if(SFM_WERROR)
    list(APPEND cxx_flags -Werror)
    list(APPEND cuda_flags -Werror=all-warnings -Xcompiler=-Werror)
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)
endfunction()
