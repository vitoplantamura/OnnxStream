cmake_minimum_required(VERSION 3.10.0)

project (sd)

set (CMAKE_CXX_STANDARD 20)

if(NOT XNNPACK_DIR)
	message(FATAL_ERROR "Please specify XNNPACK_DIR.")
endif()

if(UNIX) # includes APPLE

	include_directories("${XNNPACK_DIR}/include")
	include_directories("${XNNPACK_DIR}/build/pthreadpool-source/include")
	link_libraries("${XNNPACK_DIR}/build/libXNNPACK.a")
	link_libraries("${XNNPACK_DIR}/build/pthreadpool/libpthreadpool.a")
	link_libraries("${XNNPACK_DIR}/build/cpuinfo/libcpuinfo.a")
	link_libraries("pthread")

elseif(WIN32)

	include_directories("${XNNPACK_DIR}/include")
	include_directories("${XNNPACK_DIR}/build/pthreadpool-source/include")
	link_libraries("${XNNPACK_DIR}/build/Release/XNNPACK.lib")
	link_libraries("${XNNPACK_DIR}/build/pthreadpool/Release/pthreadpool.lib")
	link_libraries("${XNNPACK_DIR}/build/cpuinfo/Release/cpuinfo.lib")

else()

	message(FATAL_ERROR "not supported")

endif()

add_executable(sd sd.cpp onnxstream.cpp)
