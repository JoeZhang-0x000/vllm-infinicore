cmake_minimum_required(VERSION 3.16)
project(vllm_infinicore_ascend LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT INFINICORE_SOURCE OR NOT INFINICORE_REVISION OR NOT ASCEND_CANN_PACKAGE_PATH OR NOT SOC_VERSION)
  message(FATAL_ERROR "Use scripts/build_ascend.py to supply the locked source and Ascend configuration")
endif()
set(RUN_MODE "npu")
set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
set(OPS "${INFINICORE_SOURCE}/src/infiniop/ops")
ascendc_library(infinicore_ascend_kernels STATIC
  "${OPS}/swiglu/ascend/swiglu_ascend_kernel.cpp"
  "${OPS}/rope/ascend/rope_ascend_kernel.cpp"
  "${OPS}/rms_norm/ascend/cast_kernel.cpp")
target_include_directories(infinicore_ascend_kernels PRIVATE "${INFINICORE_SOURCE}/include")
set(SOURCES
  "${INFINICORE_SOURCE}/src/utils/custom_types.cc"
  "${INFINICORE_SOURCE}/src/infiniop/tensor_descriptor.cc"
  "${INFINICORE_SOURCE}/src/infiniop/operator_descriptor.cc"
  "${INFINICORE_SOURCE}/src/infiniop/devices/ascend/ascend_handle.cc"
  "${INFINICORE_SOURCE}/src/infiniop/devices/ascend/common_ascend.cc")
foreach(OP rms_norm swiglu rope embedding gemm)
  file(GLOB DEVICE_SOURCES "${OPS}/${OP}/ascend/*.cc")
  list(APPEND SOURCES "${OPS}/${OP}/operator.cc" ${DEVICE_SOURCES})
endforeach()
add_library(vllm_infinicore_ascend SHARED bridge.cpp ${SOURCES})
target_compile_definitions(vllm_infinicore_ascend PRIVATE ENABLE_ASCEND_API INFINICORE_REVISION="${INFINICORE_REVISION}")
target_include_directories(vllm_infinicore_ascend PRIVATE
  "${INFINICORE_SOURCE}/include" "${INFINICORE_SOURCE}/src"
  "${ASCEND_CANN_PACKAGE_PATH}/include" "${ASCEND_CANN_PACKAGE_PATH}/include/aclnn"
  "${ASCEND_CANN_PACKAGE_PATH}/include/aclnnop")
target_link_directories(vllm_infinicore_ascend PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(vllm_infinicore_ascend PRIVATE infinicore_ascend_kernels ascendcl nnopbase opapi runtime)
target_link_options(vllm_infinicore_ascend PRIVATE "-Wl,--no-undefined")
set_target_properties(vllm_infinicore_ascend PROPERTIES BUILD_RPATH "${ASCEND_CANN_PACKAGE_PATH}/lib64")
