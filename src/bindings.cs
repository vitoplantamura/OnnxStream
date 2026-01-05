/*

C# bindings for OnnxStream
==========================

1) Build
--------

To build the DLL/so, follow the instructions in the main README but add "-DOS_SHAREDLIB=ON" to the first CMake invocation.

2) Example
----------

This is an example of a c# program using OnnxStream. The "NativeLibrary.SetDllImportResolver" part is optional: you can
set "_libraryName" in the bindings code to the actual relative or absolute path of the DLL/so.

class Program
{
    static void Main(string[] args)
    {
        IntPtr handle = IntPtr.Zero;
        NativeLibrary.SetDllImportResolver(typeof(Program).Assembly, (name, _, _) =>
        {
            if (name == "onnxstream")
            {
                if (handle != IntPtr.Zero)
                    return handle;
                handle = NativeLibrary.Load(@"C:\Users\Vito\Desktop\test_cpp\ctypes\OnnxStream\src\build_win\Release\onnxstream.dll");
                return handle;
            }

            return IntPtr.Zero;
        });

        //     create a model instance, specifying the "prefetch" weights provider. The "prefetch" wp reads the weights files
        //     in a parallel thread, prefetching them, trying to make the next weights file required during Run already available
        using (var model = new OnnxStream.Model(0, "prefetch"))
        {
            //     during Run, write the current onnx operation to the standard output, useful for debugging
            model.OpsPrintf = true;
            //     add an intermediate result to the list of tensors that can be read with GetTensor after execution is completed
            model.AddExtraOutput("/te1/text_model/encoder/layers.11/mlp/activation_fn/Mul_output_0");
            //     load the first text encoder of SDXL Turbo. Model files are text files, so you can also specify the operations
            //     to execute directly in a string and use ReadString to load them. For an example in the WASM API, see this:
            //     https://github.com/vitoplantamura/OnnxStream/blob/af5be2d81aaa7dc7d4fd6e51c0f05809e2bb916f/examples/YOLOv8n_wasm/index.html#L411
            model.ReadFile(@"C:\Users\Vito\Desktop\stable-diffusion-xl-turbo-1.0-anyshape-onnxstream\sdxl_text_encoder_1_fp32\model.txt");
            //     add the input tensor
            model.AddTensor("input_ids", new uint[] { 1, 77 }, Enumerable.Repeat<Int64>(42, 77).ToArray());
            //     run the model
            model.Run();
            //     print the names of all the tensors that can be read with GetTensor. This includes the output tensors and all
            //     the intermediate results specified with AddExtraOutput
            foreach (string tn in model.GetAllTensorNames())
                Console.WriteLine(tn);
            //     get the output tensor
            uint[] shape;
            float[] data = model.GetTensor("out_0", out shape);
        }
    }
}

*/

using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace OnnxStream
{
    internal static class NativeApi
    {
        // _libraryName can also be a relative or absolute path to the DLL/so.
        // You can use NativeLibrary.SetDllImportResolver to specify the lib path at runtime.
        private const string _libraryName = "onnxstream";

        [StructLayout(LayoutKind.Sequential)]
        public struct GetTensorReturnLayout
        {
            public IntPtr dims_num;
            public IntPtr dims;
            public IntPtr data_num;
            public IntPtr data;
        }

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_new();

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_new_2(int threads_count, [MarshalAs(UnmanagedType.LPStr)] string wp_name);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_delete(IntPtr obj);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_read_string(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string str);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_read_file(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string fn);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_get_weights_names(IntPtr obj);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_add_weights_file(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string type, [MarshalAs(UnmanagedType.LPStr)] string name, uint size);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_add_tensor(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string type, [MarshalAs(UnmanagedType.LPStr)] string name, uint dims_num, uint[] dims);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_get_tensor(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_get_all_tensor_names(IntPtr obj);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_run(IntPtr obj);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr model_run_2(IntPtr obj);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_clear_tensors(IntPtr obj);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_set_option(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string name, uint value);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_add_extra_output(IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(_libraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void model_free_buffer(IntPtr ptr);
    }

    public class OnnxStreamError : Exception
    {
        public OnnxStreamError(string message, Exception inner = null) : base(message, inner)
        {
        }
    }

    public class Model : IDisposable
    {
        private IntPtr _modelHandle = IntPtr.Zero;
        public bool _mangleTensorNames = true;

        public Model(int threadsCount = 0, string weightsProviderName = "ram")
        {
            if (new[] { "ram", "nocache", "prefetch", "ram+nocache", "ram+prefetch" }.Contains(weightsProviderName) == false)
                throw new OnnxStreamError("Invalid weights provider name");
            _modelHandle = NativeApi.model_new_2(threadsCount, weightsProviderName);
            if (_modelHandle == IntPtr.Zero)
                throw new OnnxStreamError("Unable to create the native model object");
        }

        public void Dispose()
        {
            if (_modelHandle != IntPtr.Zero)
            {
                NativeApi.model_delete(_modelHandle);
                _modelHandle = IntPtr.Zero;
            }
        }

        public void ReadFile(string filename)
        {
            IntPtr error = NativeApi.model_read_file(_modelHandle, filename);
            if (error != IntPtr.Zero)
            {
                var errorAsStr = Marshal.PtrToStringAnsi(error);
                NativeApi.model_free_buffer(error);
                throw new OnnxStreamError(errorAsStr);
            }
        }

        public void ReadString(string str)
        {
            NativeApi.model_read_string(_modelHandle, str);
        }

        public void Run()
        {
            IntPtr error = NativeApi.model_run_2(_modelHandle);
            if (error != IntPtr.Zero)
            {
                var errorAsStr = Marshal.PtrToStringAnsi(error);
                NativeApi.model_free_buffer(error);
                throw new OnnxStreamError(errorAsStr);
            }
        }

        public static string MangleName(string name)
        {
            var finalName = new StringBuilder();

            foreach (char c in name)
            {
                if (char.IsLetterOrDigit(c))
                {
                    finalName.Append(c);
                }
                else
                {
                    string hexValue = ((int)c).ToString("X");

                    finalName.Append("_");
                    finalName.Append(hexValue);
                    finalName.Append("_");
                }
            }

            return finalName.ToString();
        }

        public static string DemangleName(string name)
        {
            string originalName = Regex.Replace(
                name,
                @"_([0-9A-Fa-f]+)_",
                match =>
                {
                    string hexValue = match.Groups[1].Value;

                    try
                    {
                        int charCode = Convert.ToInt32(hexValue, 16);
                        return ((char)charCode).ToString();
                    }
                    catch
                    {
                        return match.Value;
                    }
                }
            );

            return originalName;
        }

        public void AddTensor(string name, uint[] shape, object data)
        {
            uint els = 1;
            foreach (uint el in shape)
                els *= el;

            string type;
            if (data is float[])
            {
                if ((data as float[]).Length != els)
                    throw new OnnxStreamError("Invalid size of data wrt shape");
                type = "float32";
            }
            else if (data is Int64[])
            {
                if ((data as Int64[]).Length != els)
                    throw new OnnxStreamError("Invalid size of data wrt shape");
                type = "int64";
            }
            else
            {
                throw new OnnxStreamError("Invalid type of data");
            }

            IntPtr ptr = NativeApi.model_add_tensor(_modelHandle, type, _mangleTensorNames ? MangleName(name) : name, (uint)shape.Length, shape);

            if (data is float[])
                Marshal.Copy(data as float[], 0, ptr, (int)els);
            else
                Marshal.Copy(data as Int64[], 0, ptr, (int)els);
        }

        public string[] GetAllTensorNames()
        {
            IntPtr ret = NativeApi.model_get_all_tensor_names(_modelHandle);
            var retAsStr = Marshal.PtrToStringAnsi(ret);
            NativeApi.model_free_buffer(ret);
            string[] retAsArr = retAsStr.Split('|');
            if (_mangleTensorNames)
            {
                for (int i = 0; i < retAsArr.Length; i++)
                    retAsArr[i] = DemangleName(retAsArr[i]);
            }
            return retAsArr;
        }

        public float[] GetTensor(string name, out uint[] shape)
        {
            IntPtr ret = NativeApi.model_get_tensor(_modelHandle, _mangleTensorNames ? MangleName(name) : name);
            if (ret == IntPtr.Zero)
            {
                shape = new uint[0];
                return null;
            }

            var retAsStr = Marshal.PtrToStructure<NativeApi.GetTensorReturnLayout>(ret);
            NativeApi.model_free_buffer(ret);

            float[] data = new float[retAsStr.data_num];
            Marshal.Copy(retAsStr.data, data, 0, data.Length);

            IntPtr[] shapeAsIp = new IntPtr[retAsStr.dims_num];
            Marshal.Copy(retAsStr.dims, shapeAsIp, 0, shapeAsIp.Length);
            shape = Array.ConvertAll(shapeAsIp, v => (uint)v);
            return data;
        }

        public void ClearTensors()
        {
            NativeApi.model_clear_tensors(_modelHandle);
        }

        private void SetOption(string name, bool val)
        {
            NativeApi.model_set_option(_modelHandle, name, (uint)(val ? 1 : 0));
        }

        public bool UseFp16Arithmetic { set => SetOption("use_fp16_arithmetic", value); }
        public bool UseUint8Qdq { set => SetOption("use_uint8_qdq", value); }
        public bool UseUint8Arithmetic { set => SetOption("use_uint8_arithmetic", value); }
        public bool FuseOpsInAttention { set => SetOption("fuse_ops_in_attention", value); }
        public bool ForceFp16Storage { set => SetOption("force_fp16_storage", value); }
        public bool SupportDynamicShapes { set => SetOption("support_dynamic_shapes", value); }
        public bool UseOpsCache { set => SetOption("use_ops_cache", value); }
        public bool UseScaledDpAttnOp { set => SetOption("use_scaled_dp_attn_op", value); }
        public bool UseNextOpCache { set => SetOption("use_next_op_cache", value); }
        public bool OpsPrintf { set => SetOption("ops_printf", value); }
        public bool OpsTimesPrintf { set => SetOption("ops_times_printf", value); }
        public bool UseNchwConvs { set => SetOption("use_nchw_convs", value); }

        public void AddExtraOutput(string name)
        {
            NativeApi.model_add_extra_output(_modelHandle, _mangleTensorNames ? MangleName(name) : name);
        }
    }
}
