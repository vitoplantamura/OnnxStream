cmake_minimum_required(VERSION 3.10.0)

option(MAX_SPEED "Enable aggressive speed optimizations" ON)
option(OS_LLM "Select the LLM example instead of the SD example" OFF)
option(OS_CUDA "Enable GPU acceleration with CUDA" OFF)
option(OS_SHAREDLIB "Create a shared library instead of an executable. CMake XNNPACK with -DCMAKE_POSITION_INDEPENDENT_CODE=ON" OFF)
option(USE_LIBPNG "Use libpng for saving compressed images" OFF)
option(USE_LIBJPEGTURBO "Use libjpeg-turbo for saving compressed images" OFF)
set(XNNPACK_DIR "" CACHE PATH "Path to pre-built XNNPACK.")

if(NOT XNNPACK_SOURCE OR NOT XNNPACK_BINARY)
	if(XNNPACK_DIR)
		set(XNNPACK_SOURCE ${XNNPACK_DIR})
		set(XNNPACK_BINARY ${XNNPACK_DIR}/build)
	else()
		set(XNNPACK_SOURCE ${CMAKE_BINARY_DIR}/XNNPACK-source)
		set(XNNPACK_BINARY ${CMAKE_BINARY_DIR}/XNNPACK)

		message(STATUS "XNNPACK_DIR and XNNPACK_SOURCE/XNNPACK_BINARY not set. Using ExternalProject to download XNNPACK.")

		IF(POLICY CMP0135)
		CMAKE_POLICY(SET CMP0135 NEW)
		ENDIF()

		include(ExternalProject)
		ExternalProject_Add(xnnpack_build
			URL https://github.com/google/XNNPACK/archive/f64bbdddff56cca640ade7a9770e70b471eb5630.zip
			URL_HASH SHA256=745d8b73794cfe466064004f7ede01d9e9930774a392331d447727d11cd62e08
			SOURCE_DIR "${XNNPACK_SOURCE}"
			BINARY_DIR "${XNNPACK_BINARY}"
			CMAKE_ARGS
				-DCMAKE_BUILD_TYPE=Release
				-DXNNPACK_BUILD_TESTS=OFF
				-DXNNPACK_BUILD_BENCHMARKS=OFF
				-DCMAKE_POSITION_INDEPENDENT_CODE=${OS_SHAREDLIB}
			INSTALL_COMMAND ""
		)

		ExternalProject_Get_Property(xnnpack_build source_dir binary_dir)

		set(XNNPACK_BUILD_DEPENDENCY xnnpack_build)
	endif()
endif()

set(ONNXSTREAM_PROJECT_NAME sd)
if(OS_LLM)
	set(ONNXSTREAM_PROJECT_NAME llm)
elseif(OS_SHAREDLIB)
	set(ONNXSTREAM_PROJECT_NAME onnxstream)
endif()

project (${ONNXSTREAM_PROJECT_NAME})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set (CMAKE_CXX_STANDARD 20)

if(NOT OS_SHAREDLIB)
	add_executable(${ONNXSTREAM_PROJECT_NAME} ${ONNXSTREAM_PROJECT_NAME}.cpp onnxstream.cpp)
else()
	add_library(${ONNXSTREAM_PROJECT_NAME} SHARED exports.cpp onnxstream.cpp)
endif()

if(XNNPACK_BUILD_DEPENDENCY)
	add_dependencies(${ONNXSTREAM_PROJECT_NAME} ${XNNPACK_BUILD_DEPENDENCY})
endif()

if(UNIX) # includes APPLE and ANDROID (ie Termux)

	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_SOURCE}/include")
	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/pthreadpool-source/include")
	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/cpuinfo-source/include")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/libXNNPACK.a")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/libmicrokernels-prod.a")
	if(EXISTS "${XNNPACK_BINARY}/kleidiai/libkleidiai.a")
		target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/kleidiai/libkleidiai.a")
	endif()
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/pthreadpool/libpthreadpool.a")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/cpuinfo/libcpuinfo.a")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "pthread")

	if(USE_LIBPNG)
		target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "png")
		target_compile_options(${ONNXSTREAM_PROJECT_NAME} PRIVATE -DUSE_LIBPNG)
	endif(USE_LIBPNG)

	if(USE_LIBJPEGTURBO)
		target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "jpeg")
		target_compile_options(${ONNXSTREAM_PROJECT_NAME} PRIVATE -DUSE_LIBJPEGTURBO)
	endif(USE_LIBJPEGTURBO)

	if(ANDROID)
		target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "log")
	endif()

	if(MAX_SPEED)
		target_compile_options(${ONNXSTREAM_PROJECT_NAME} PRIVATE -O3 -march=native)
	endif()

elseif(WIN32)

	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_SOURCE}/include")
	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/pthreadpool-source/include")
	target_include_directories(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/cpuinfo-source/include")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/Release/XNNPACK.lib")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/Release/microkernels-prod.lib")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/pthreadpool/Release/pthreadpool.lib")
	target_link_libraries(${ONNXSTREAM_PROJECT_NAME} PRIVATE "${XNNPACK_BINARY}/cpuinfo/Release/cpuinfo.lib")

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
