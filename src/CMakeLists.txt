cmake_minimum_required(VERSION 3.10.0)

option(MAX_SPEED "Enable aggressive speed optimizations" OFF)
option(OS_LLM "Select the LLM example instead of the SD example" OFF)
option(OS_CUDA "Enable GPU acceleration with CUDA" OFF)

if(NOT XNNPACK_DIR)
	message(FATAL_ERROR "Please specify XNNPACK_DIR.")
endif()

set(ONNXSTREAM_PROJECT_NAME sd)
if(OS_LLM)
	set(ONNXSTREAM_PROJECT_NAME llm)
endif()

project (${ONNXSTREAM_PROJECT_NAME})

set (CMAKE_CXX_STANDARD 20)

add_executable(${ONNXSTREAM_PROJECT_NAME} ${ONNXSTREAM_PROJECT_NAME}.cpp onnxstream.cpp)

if(UNIX) # includes APPLE and ANDROID (ie Termux)

	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/include")
	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/pthreadpool-source/include")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/libXNNPACK.a")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/pthreadpool/libpthreadpool.a")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/cpuinfo/libcpuinfo.a")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "pthread")

	if(ANDROID)
		target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "log")
	endif()

	if(MAX_SPEED)
		target_compile_options(${ONNXSTREAM_PROJECT_NAME} PRIVATE -O3 -march=native)
	endif()

elseif(WIN32)

	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/include")
	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/pthreadpool-source/include")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/Release/XNNPACK.lib")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/pthreadpool/Release/pthreadpool.lib")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_DIR}/build/cpuinfo/Release/cpuinfo.lib")

	if(MAX_SPEED)
		target_compile_options(${ONNXSTREAM_PROJECT_NAME} PRIVATE /GL /Ot)
	endif()

else()

	message(FATAL_ERROR "not supported")

endif()

if(OS_CUDA)

	find_package(CUDAToolkit REQUIRED)

	set_target_properties(${ONNXSTREAM_PROJECT_NAME}
	  PROPERTIES
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)

	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE
	  PRIVATE
		CUDA::cublas
		CUDA::cudart_static)

	target_compile_definitions(${ONNXSTREAM_PROJECT_NAME} PRIVATE ONNXSTREAM_CUDA=1)

endif()
