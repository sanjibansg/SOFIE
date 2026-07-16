if(NOT DEFINED GENERATED_DIR)
  message(FATAL_ERROR "GENERATED_DIR is required")
endif()

function(check_externalized_storage model storage length)
  set(header "${GENERATED_DIR}/${model}_FromONNX_GPU_ALPAKA.hxx")
  set(weights "${GENERATED_DIR}/${model}_FromONNX_GPU_ALPAKA.dat")

  if(NOT EXISTS "${header}")
    message(FATAL_ERROR "Generated quantized header is missing: ${header}")
  endif()
  if(NOT EXISTS "${weights}")
    message(FATAL_ERROR "Generated quantized weight file is missing: ${weights}")
  endif()

  file(READ "${header}" header_text)
  file(READ "${weights}" weight_text)

  string(FIND "${header_text}" "fTensor_${storage} =" embedded_storage)
  if(NOT embedded_storage EQUAL -1)
    message(FATAL_ERROR
      "Quantized storage ${storage} is embedded as a generated C++ literal")
  endif()

  string(FIND "${header_text}" "std::vector<int8_t> tensor_${storage}(${length});" loaded_storage)
  if(loaded_storage EQUAL -1)
    message(FATAL_ERROR
      "Generated code does not allocate file-loaded int8 storage ${storage}")
  endif()

  string(FIND "${weight_text}" "tensor_${storage} ${length}" serialized_storage)
  if(serialized_storage EQUAL -1)
    message(FATAL_ERROR
      "Generated weight file does not contain int8 storage ${storage}")
  endif()

  string(FIND "${header_text}" "_s11_packed_cpu_storage" cpu_storage_header)
  string(FIND "${weight_text}" "_s11_packed_cpu_storage" cpu_storage_weights)
  if(NOT cpu_storage_header EQUAL -1 OR NOT cpu_storage_weights EQUAL -1)
    message(FATAL_ERROR
      "ALPAKA output for ${model} contains CPU-only packed quantized storage")
  endif()
endfunction()

function(check_storage_absent model storage)
  file(READ "${GENERATED_DIR}/${model}_FromONNX_GPU_ALPAKA.hxx" header_text)
  file(READ "${GENERATED_DIR}/${model}_FromONNX_GPU_ALPAKA.dat" weight_text)
  string(FIND "${header_text}" "${storage}" storage_header)
  string(FIND "${weight_text}" "${storage}" storage_weights)
  if(NOT storage_header EQUAL -1 OR NOT storage_weights EQUAL -1)
    message(FATAL_ERROR "ALPAKA output for ${model} contains unused storage ${storage}")
  endif()
endfunction()

function(check_source_absent model source)
  file(READ "${GENERATED_DIR}/${model}_FromONNX_GPU_ALPAKA.hxx" header_text)
  file(READ "${GENERATED_DIR}/${model}_FromONNX_GPU_ALPAKA.dat" weight_text)
  string(FIND "${header_text}" "deviceBuf_${source} =" source_device_buffer)
  string(FIND "${header_text}" "tensor_${source}(" source_host_buffer)
  string(FIND "${weight_text}" "tensor_${source} " source_weights)
  if(NOT source_device_buffer EQUAL -1 OR NOT source_host_buffer EQUAL -1 OR NOT source_weights EQUAL -1)
    message(FATAL_ERROR "Deployment artifacts for ${model} retain superseded source tensor ${source}")
  endif()
endfunction()

check_externalized_storage(
  QONNX_QuantGemm weight_fp_quantized_plain_device_storage 2048)
check_externalized_storage(
  QONNX_QuantMatMul weight_fp_quantized_transposed_device_storage 2048)
check_externalized_storage(
  QONNX_QuantMatMul_Padded weight_fp_quantized_transposed_padded_device_storage 5120)
check_externalized_storage(
  ONNX_QDQ_QuantGemm weight_dequantized_quantized_plain_device_storage 4096)
check_externalized_storage(
  ONNX_QDQ_QuantMatMul weight_dequantized_quantized_transposed_device_storage 4096)
check_storage_absent(
  QONNX_QuantMatMul_Padded weight_fp_quantized_transposed_device_storage)
check_source_absent(QONNX_QuantGemm weight_fp)
check_source_absent(QONNX_QuantMatMul weight_fp)
check_source_absent(QONNX_QuantMatMul_Padded weight_fp)
check_source_absent(ONNX_QDQ_QuantGemm weight_dequantized)
check_source_absent(ONNX_QDQ_QuantGemm weight_q)
check_source_absent(ONNX_QDQ_QuantMatMul weight_dequantized)
check_source_absent(ONNX_QDQ_QuantMatMul weight_q)

set(binary_weights "${GENERATED_DIR}/QONNX_QuantGemm_Binary_FromONNX_GPU_ALPAKA.bin")
if(NOT EXISTS "${binary_weights}")
  message(FATAL_ERROR "Generated quantized binary weight file is missing: ${binary_weights}")
endif()
file(SIZE "${binary_weights}" binary_size)
if(NOT binary_size LESS 10000)
  message(FATAL_ERROR "Quantized binary fixture is unexpectedly large: ${binary_size} bytes")
endif()
file(READ "${binary_weights}" binary_magic OFFSET 0 LIMIT 8 HEX)
if(NOT binary_magic STREQUAL "534f464945574231")
  message(FATAL_ERROR "Quantized binary fixture has an invalid SOFIEWB1 magic header")
endif()
