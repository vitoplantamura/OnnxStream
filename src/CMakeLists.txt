cmake_minimum_required(VERSION 3.10.0)

project (sd)

set (CMAKE_CXX_STANDARD 20)

option(MAX_SPEED "Enable aggressive speed optimizations" OFF)

if(NOT XNNPACK_DIR)
	message(FATAL_ERROR "Please specify XNNPACK_DIR.")
endif()

add_executable(sd sd.cpp onnxstream.cpp)

if(UNIX) # includes APPLE and ANDROID (ie Termux)

	target_include_directories(sd PRIVATE "${XNNPACK_DIR}/include")
	target_include_directories(sd PRIVATE "${XNNPACK_DIR}/build/pthreadpool-source/include")
	target_link_libraries(sd PRIVATE "${XNNPACK_DIR}/build/libXNNPACK.a")
	target_link_libraries(sd PRIVATE "${XNNPACK_DIR}/build/pthreadpool/libpthreadpool.a")
	target_link_libraries(sd PRIVATE "${XNNPACK_DIR}/build/cpuinfo/libcpuinfo.a")
	target_link_libraries(sd PRIVATE "pthread")

	if(ANDROID)
		target_link_libraries(sd PRIVATE "log")
	endif()

	if(MAX_SPEED)
		target_compile_options(sd PRIVATE -O3 -march=native)
	endif()

elseif(WIN32)

	target_include_directories(sd PRIVATE "${XNNPACK_DIR}/include")
	target_include_directories(sd PRIVATE "${XNNPACK_DIR}/build/pthreadpool-source/include")
	target_link_libraries(sd PRIVATE "${XNNPACK_DIR}/build/Release/XNNPACK.lib")
	target_link_libraries(sd PRIVATE "${XNNPACK_DIR}/build/pthreadpool/Release/pthreadpool.lib")
	target_link_libraries(sd PRIVATE "${XNNPACK_DIR}/build/cpuinfo/Release/cpuinfo.lib")

	if(MAX_SPEED)
		target_compile_options(sd PRIVATE /GL /Ot)
	endif()

else()

	message(FATAL_ERROR "not supported")

endif()
