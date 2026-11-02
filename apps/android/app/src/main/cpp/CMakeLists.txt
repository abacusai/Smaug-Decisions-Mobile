cmake_minimum_required(VERSION 3.22)
project(decide_jni C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Smaug-Decisions backend: litert (default; LiteRT-LM C API with our harness patch, prebuilt per ABI in LITERT_DIR) or llamacpp (the earlier
# llama.cpp path, kept as a fallback flavor: -PbigBackend=llamacpp). Both share decide_jni-style JNI glue; only one is built.
set(DECIDE_BIG_BACKEND "litert" CACHE STRING "litert or llamacpp")

if(DECIDE_BIG_BACKEND STREQUAL "litert")
    if(NOT LITERT_DIR)
        message(FATAL_ERROR "pass -DLITERT_DIR=<dir with include/c/*.h and android_arm64/ or linux_x86_64/ liblitert-lm.so>")
    endif()
    if(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(LITERT_ABI android_arm64)
        else()
            message(FATAL_ERROR "the LiteRT-LM runtime is built for arm64-v8a only (got ${ANDROID_ABI})")
        endif()
    else()
        set(LITERT_ABI linux_x86_64)
    endif()
    add_library(litert_lm SHARED IMPORTED)
    # liblitert-lm.so has no SONAME: IMPORTED_NO_SONAME makes the link record "liblitert-lm.so", not the build path
    set_target_properties(litert_lm PROPERTIES IMPORTED_LOCATION ${LITERT_DIR}/${LITERT_ABI}/liblitert-lm.so IMPORTED_NO_SONAME TRUE)
    add_library(decide_litert_jni SHARED ${CMAKE_CURRENT_LIST_DIR}/litert/DecideShim.c ${CMAKE_CURRENT_LIST_DIR}/litert/litert_jni.c)
    target_include_directories(decide_litert_jni PRIVATE ${LITERT_DIR}/include ${LITERT_DIR}/include/c ${CMAKE_CURRENT_LIST_DIR}/litert)
    if(ANDROID)
        target_link_libraries(decide_litert_jni PRIVATE litert_lm log)
    else()
        find_package(JNI REQUIRED COMPONENTS JVM)
        target_include_directories(decide_litert_jni PRIVATE ${JNI_INCLUDE_DIRS})
        target_link_libraries(decide_litert_jni PRIVATE litert_lm)
        set_target_properties(decide_litert_jni PROPERTIES BUILD_RPATH ${LITERT_DIR}/${LITERT_ABI})
    endif()
    # optional CLI over the same C path (litert/litert_replay.c), for checking a cross-compiled runtime without a device
    option(DECIDE_BUILD_REPLAY "build the replay CLI" OFF)
    if(DECIDE_BUILD_REPLAY)
        add_executable(litert_replay ${CMAKE_CURRENT_LIST_DIR}/litert/DecideShim.c ${CMAKE_CURRENT_LIST_DIR}/litert/litert_replay.c)
        target_include_directories(litert_replay PRIVATE ${LITERT_DIR}/include ${LITERT_DIR}/include/c ${CMAKE_CURRENT_LIST_DIR}/litert)
        target_link_libraries(litert_replay PRIVATE litert_lm)
        set_target_properties(litert_replay PROPERTIES BUILD_RPATH ${LITERT_DIR}/${LITERT_ABI})
    endif()
    return()
endif()

if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "pass -DLLAMA_CPP_DIR=<llama.cpp checkout>")
endif()

# llama.cpp + libmtmd only (no tools/common/server), everything static into one libdecide_jni.so
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
# static llama.cpp/ggml linked into a shared JNI library: needs PIC on desktop x86_64 (the NDK defaults to PIC already)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(ANDROID)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
else()
    # host build (desktop JVM tests: see run_host_e2e.sh): tune for the build machine
    set(GGML_NATIVE ON CACHE BOOL "")
endif()
if(ANDROID_ABI STREQUAL "x86_64")
    # emulator / Chromebook ABI: x86-64-v3 (AVX2 + FMA + F16C) is what current x86 Android emulators expose
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()
if(ANDROID_ABI STREQUAL "arm64-v8a")
    # armv8.2-a + dotprod + fp16 covers Snapdragon 8xx / Tensor / Dimensity flagships from ~2019 on (i8mm for Q4_0 repack on newer)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()
# GPU (optional, untested here): -DGGML_VULKAN=ON (needs glslc from the NDK/SDK) or -DGGML_OPENCL=ON for Adreno

add_subdirectory(${LLAMA_CPP_DIR} build-llama)

# the C core is shared with the Python path (llama.cpp/llama_decide.py); one copy, in android/llama.cpp
set(DECIDE_CORE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../llama.cpp)

add_library(decide_jni SHARED
    ${DECIDE_CORE_DIR}/decide_core.c
    ${CMAKE_CURRENT_LIST_DIR}/decide_jni.c)
target_include_directories(decide_jni PRIVATE ${DECIDE_CORE_DIR} ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/ggml/include ${LLAMA_CPP_DIR}/tools/mtmd)
if(ANDROID)
    target_link_libraries(decide_jni PRIVATE mtmd llama ggml android log)
else()
    find_package(JNI REQUIRED COMPONENTS JVM)   # headers only; headless JDKs have no AWT
    target_include_directories(decide_jni PRIVATE ${JNI_INCLUDE_DIRS})
    target_link_libraries(decide_jni PRIVATE mtmd llama ggml)
endif()

# optional: decide_replay CLI (../llama.cpp/decide_replay.c) to check a cross-compiled build without a device, e.g. the arm64 build
# under qemu-aarch64 (llama.cpp/qemu_arm64_check.py). Configure with -DDECIDE_BUILD_REPLAY=ON (and -DANDROID_STL=c++_static).
option(DECIDE_BUILD_REPLAY "build the decide_replay CLI" OFF)
if(DECIDE_BUILD_REPLAY)
    add_executable(decide_replay ${DECIDE_CORE_DIR}/decide_core.c ${DECIDE_CORE_DIR}/decide_replay.c)
    target_include_directories(decide_replay PRIVATE ${DECIDE_CORE_DIR} ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/ggml/include ${LLAMA_CPP_DIR}/tools/mtmd)
    target_link_libraries(decide_replay PRIVATE mtmd llama ggml)
    if(ANDROID)
        target_link_options(decide_replay PRIVATE -static)   # static bionic: runs under qemu-aarch64 without Android's linker
    endif()
endif()
