'''

Python bindings for OnnxStream
==============================

1) Build
--------

In order to build the DLL/so, follow the instructions in the main README for building the Stable Diffusion executable, and:

- If on Linux, when building XNNPACK, add "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" to the first CMake invocation.

- When building the Stable Diffusion application, add "-DOS_SHAREDLIB=ON" to the first CMake invocation.

2) Example
----------

#     create a Model instance, specifying the "prefetch" weights provider. The "prefetch" wp reads the weights files
#     in a parallel thread, prefetching them, trying to make the next weights file required during Run already available
with Model(library_path="./build/libonnxstream.so", threads_count=0, weights_provider_name="prefetch") as model:
    #     during Run, write the current onnx operation to the standard output, useful for debugging
    model.set_ops_printf(True)
    #     add an intermediate result to the list of tensors that can be read with GetTensor after execution is completed
    model.add_extra_output("/te1/text_model/encoder/layers.11/mlp/activation_fn/Mul_output_0")
    #     load the first text encoder of SDXL Turbo. Model files are text files, so you can also specify the operations
    #     to execute directly in a string and use ReadString to load them. For an example in the WASM API, see this:
    #     https://github.com/vitoplantamura/OnnxStream/blob/af5be2d81aaa7dc7d4fd6e51c0f05809e2bb916f/examples/YOLOv8n_wasm/index.html#L411
    model.read_file("/home/vito/Desktop/stable-diffusion-xl-turbo-1.0-anyshape-onnxstream/sdxl_text_encoder_1_fp32/model.txt")
    #     add the input tensor
    model.add_tensor("input_ids", np.full((1, 77), 42, dtype=np.int64))
    #     run the model
    model.run()
    #     print the names of all the tensors that can be read with GetTensor. This includes the output tensors and all
    #     the intermediate results specified with AddExtraOutput
    tensor_names = model.get_all_tensor_names()
    for tn in tensor_names:
        print(f"- {tn}")
    #     get the output tensor
    output_data, output_shape = model.get_tensor("out_0")

'''

import ctypes
import re
import numpy as np
import os
import sys
from typing import List, Tuple, Union

class OnnxStreamError(Exception):
    pass

class GetTensorReturnLayout(ctypes.Structure):
    _fields_ = [
        ("dims_num", ctypes.c_void_p),
        ("dims", ctypes.c_void_p),
        ("data_num", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
    ]

class Model:
    def __init__(self, library_path: str, threads_count: int = 0, weights_provider_name: str = "prefetch"):
        self._lib = ctypes.CDLL(library_path)
        self._setup_prototypes()
        self.mangle_tensor_names = True

        valid_providers = ["ram", "nocache", "prefetch", "ram+nocache", "ram+prefetch"]
        if weights_provider_name not in valid_providers:
            raise OnnxStreamError(f"Invalid weights provider name: {weights_provider_name}")

        wp_name_bytes = weights_provider_name.encode('utf-8')
        self._model_handle = self._lib.model_new_2(threads_count, wp_name_bytes)

        if not self._model_handle:
            raise OnnxStreamError("Unable to create the native model object")

    def _setup_prototypes(self):
        self._lib.model_new_2.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self._lib.model_new_2.restype = ctypes.c_void_p

        self._lib.model_delete.argtypes = [ctypes.c_void_p]
        self._lib.model_delete.restype = None

        self._lib.model_read_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.model_read_file.restype = ctypes.c_void_p

        self._lib.model_read_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.model_read_string.restype = None

        self._lib.model_run_2.argtypes = [ctypes.c_void_p]
        self._lib.model_run_2.restype = ctypes.c_void_p

        self._lib.model_add_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        self._lib.model_add_tensor.restype = ctypes.c_void_p

        self._lib.model_get_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.model_get_tensor.restype = ctypes.c_void_p

        self._lib.model_get_all_tensor_names.argtypes = [ctypes.c_void_p]
        self._lib.model_get_all_tensor_names.restype = ctypes.c_void_p

        self._lib.model_clear_tensors.argtypes = [ctypes.c_void_p]
        self._lib.model_clear_tensors.restype = None

        self._lib.model_set_option.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self._lib.model_set_option.restype = None

        self._lib.model_add_extra_output.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.model_add_extra_output.restype = None

        self._lib.model_free_buffer.argtypes = [ctypes.c_void_p]
        self._lib.model_free_buffer.restype = None

    def _handle_error(self, error_ptr: int):
        if error_ptr:
            error_msg = ctypes.cast(error_ptr, ctypes.c_char_p).value.decode('utf-8')
            self._lib.model_free_buffer(error_ptr)
            raise OnnxStreamError(error_msg)

    def close(self):
        if self._model_handle:
            self._lib.model_delete(self._model_handle)
            self._model_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_file(self, filename: str):
        error_ptr = self._lib.model_read_file(self._model_handle, filename.encode('utf-8'))
        self._handle_error(error_ptr)

    def read_string(self, model_string: str):
        self._lib.model_read_string(self._model_handle, model_string.encode('utf-8'))

    def run(self):
        error_ptr = self._lib.model_run_2(self._model_handle)
        self._handle_error(error_ptr)
        
    def add_tensor(self, name: str, data: np.ndarray):
        shape = data.shape
        els = data.size

        if data.dtype == np.float32:
            dtype_str = "float32"
        elif data.dtype == np.int64:
            dtype_str = "int64"
        else:
            raise OnnxStreamError(f"Unsupported data type: {data.dtype}")

        final_name = self.mangle_name(name) if self.mangle_tensor_names else name
        
        shape_array = (ctypes.c_uint * len(shape))(*shape)
        
        ptr = self._lib.model_add_tensor(
            self._model_handle,
            dtype_str.encode('utf-8'),
            final_name.encode('utf-8'),
            len(shape),
            shape_array
        )

        source_ptr = data.ctypes.data
        ctypes.memmove(ptr, source_ptr, data.nbytes)
        
    def get_tensor(self, name: str) -> Tuple[np.ndarray, List[int]]:
        final_name = self.mangle_name(name) if self.mangle_tensor_names else name
        ret_ptr = self._lib.model_get_tensor(self._model_handle, final_name.encode('utf-8'))

        if not ret_ptr:
            return None

        ret_struct = ctypes.cast(ret_ptr, ctypes.POINTER(GetTensorReturnLayout)).contents

        data_num = ret_struct.data_num
        dims_num = ret_struct.dims_num

        dims_ptr = ctypes.cast(ret_struct.dims, ctypes.POINTER(ctypes.c_size_t))
        shape = [dims_ptr[i] for i in range(dims_num)]

        data_ptr = ctypes.cast(ret_struct.data, ctypes.POINTER(ctypes.c_float))
        data = np.ctypeslib.as_array(data_ptr, shape=(data_num,)).copy()
        
        self._lib.model_free_buffer(ret_ptr)

        return data.reshape(shape), shape

    def get_all_tensor_names(self) -> List[str]:
        ret_ptr = self._lib.model_get_all_tensor_names(self._model_handle)
        if not ret_ptr:
            return []
            
        ret_as_str = ctypes.cast(ret_ptr, ctypes.c_char_p).value.decode('utf-8')
        self._lib.model_free_buffer(ret_ptr)
        
        names = ret_as_str.split('|')
        
        if self.mangle_tensor_names:
            return [self.demangle_name(name) for name in names]
        return names

    def clear_tensors(self):
        self._lib.model_clear_tensors(self._model_handle)

    def add_extra_output(self, name: str):
        final_name = self.mangle_name(name) if self.mangle_tensor_names else name
        self._lib.model_add_extra_output(self._model_handle, final_name.encode('utf-8'))

    def _set_option(self, name: str, value: bool):
        self._lib.model_set_option(self._model_handle, name.encode('utf-8'), 1 if value else 0)

    def set_use_fp16_arithmetic(self, value: bool): self._set_option("use_fp16_arithmetic", value)
    def set_use_uint8_qdq(self, value: bool): self._set_option("use_uint8_qdq", value)
    def set_use_uint8_arithmetic(self, value: bool): self._set_option("use_uint8_arithmetic", value)
    def set_fuse_ops_in_attention(self, value: bool): self._set_option("fuse_ops_in_attention", value)
    def set_force_fp16_storage(self, value: bool): self._set_option("force_fp16_storage", value)
    def set_support_dynamic_shapes(self, value: bool): self._set_option("support_dynamic_shapes", value)
    def set_use_ops_cache(self, value: bool): self._set_option("use_ops_cache", value)
    def set_use_scaled_dp_attn_op(self, value: bool): self._set_option("use_scaled_dp_attn_op", value)
    def set_use_next_op_cache(self, value: bool): self._set_option("use_next_op_cache", value)
    def set_ops_printf(self, value: bool): self._set_option("ops_printf", value)
    def set_ops_times_printf(self, value: bool): self._set_option("ops_times_printf", value)
    def set_use_nchw_convs(self, value: bool): self._set_option("use_nchw_convs", value)

    @staticmethod
    def mangle_name(name: str) -> str:
        final_name = []
        for char in name:
            if char.isalnum():
                final_name.append(char)
            else:
                final_name.append(f"_{ord(char):X}_")
        return "".join(final_name)

    @staticmethod
    def demangle_name(name: str) -> str:
        def repl(match):
            hex_value = match.group(1)
            try:
                char_code = int(hex_value, 16)
                return chr(char_code)
            except (ValueError, TypeError):
                return match.group(0)
        
        return re.sub(r"_([0-9A-Fa-f]+)_", repl, name)
