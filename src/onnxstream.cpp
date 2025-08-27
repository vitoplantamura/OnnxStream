#include "onnxstream.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <ostream>
#include <random>
#include <functional>
#include <unordered_map>

#include <xnnpack.h>

static_assert(TENSOR_VECTOR_EXTRA_BYTES == XNN_EXTRA_BYTES);

#if ONNXSTREAM_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace onnxstream {

void* custom_aligned_alloc(std::size_t alignment, std::size_t size) // https://stackoverflow.com/questions/32133203/what-can-i-use-instead-of-stdaligned-alloc-in-ms-visual-studio-2013
{
    if (alignment < alignof(void*)) {
        alignment = alignof(void*);
    }
    std::size_t space = size + alignment - 1;
    void* allocated_mem = ::operator new(space + sizeof(void*));
    void* aligned_mem = static_cast<void*>(static_cast<char*>(allocated_mem) + sizeof(void*));
    if (!std::align(alignment, size, aligned_mem, space))
        throw std::runtime_error("custom_aligned_alloc: std::align error.");
    *(static_cast<void**>(aligned_mem) - 1) = allocated_mem;
    return aligned_mem;
}

void custom_aligned_free(void* p) noexcept
{
    ::operator delete(*(static_cast<void**>(p) - 1));
}

#if ONNXSTREAM_CUDA

#define CUDA_ERRCHECK(ans) { cuda_assert((ans), __FILE__, __LINE__); }
static inline void cuda_assert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("cuda_assert: ") + cudaGetErrorString(code) + " in " + file + ":" + std::to_string(line));
}

static inline const char* cublasGetErrorString_(const cublasStatus_t err)
{
    switch (err) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "unknown error";
    }
}
#define CUBLAS_ERRCHECK(ans) { cublas_assert((ans), __FILE__, __LINE__); }
static inline void cublas_assert(cublasStatus_t code, const char* file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cublas_assert: ") + cublasGetErrorString_(code) + " in " + file + ":" + std::to_string(line));
}

class CublasOps
{
private:

    struct DBuffer
    {
        void* m_d_ptr = nullptr;
        size_t m_size = 0;
        bool m_used = false;
    };

    struct PendingOp
    {
        void* m_h_buffer = nullptr;
        cudaStream_t m_stream = nullptr;
        std::vector<void*> m_d_buffers;
    };

    cublasHandle_t m_cublas_handle = nullptr;
    uint64_t m_vram_used = 0;
    std::vector<DBuffer> m_d_buffers;
    std::vector<cudaStream_t> m_streams;
    size_t m_streams_pos = 0;
    cudaEvent_t m_sync_event = nullptr;
    std::vector<PendingOp> m_pending_ops;

public:

    CudaOptions m_cuda_options;

public:

    CublasOps()
    {
    }
    ~CublasOps()
    {
        if (m_sync_event)
            CUDA_ERRCHECK(cudaEventDestroy(m_sync_event));

        for (cudaStream_t stream : m_streams)
            CUDA_ERRCHECK(cudaStreamDestroy(stream));

        if (m_cublas_handle)
            CUBLAS_ERRCHECK(cublasDestroy(m_cublas_handle));

        for (auto& d_buffer : m_d_buffers)
            if (d_buffer.m_d_ptr)
                CUDA_ERRCHECK(cudaFree(d_buffer.m_d_ptr));
    }

    cublasHandle_t get_cublas_handle()
    {
        if (!m_cublas_handle)
            CUBLAS_ERRCHECK(cublasCreate(&m_cublas_handle));
        return m_cublas_handle;
    }

    cudaStream_t get_next_stream()
    {
        if (!m_streams.size())
        {
            for (size_t i = 0; i < 8; i++)
            {
                cudaStream_t stream = nullptr;
                CUDA_ERRCHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                m_streams.push_back(stream);
            }
        }

        return m_streams[m_streams_pos++ % m_streams.size()];
    }

    cudaEvent_t get_sync_event()
    {
        if (!m_sync_event)
            CUDA_ERRCHECK(cudaEventCreateWithFlags(&m_sync_event, cudaEventBlockingSync | cudaEventDisableTiming));
        return m_sync_event;
    }

    void* get_d_buffer(size_t size)
    {
        DBuffer* d_buffer = nullptr;
        for (auto& item : m_d_buffers)
            if (!item.m_used)
            {
                item.m_used = true;
                d_buffer = &item;
                break;
            }

        if (!d_buffer)
        {
            void* d_mem = nullptr;
            CUDA_ERRCHECK(cudaMalloc(&d_mem, size));
            m_d_buffers.emplace_back(d_mem, size, true);
            return d_mem;
        }
        else if (size <= d_buffer->m_size)
        {
            return d_buffer->m_d_ptr;
        }
        else
        {
            CUDA_ERRCHECK(cudaFree(d_buffer->m_d_ptr));
            d_buffer->m_d_ptr = nullptr;
            CUDA_ERRCHECK(cudaMalloc(&d_buffer->m_d_ptr, size));
            d_buffer->m_size = size;
            return d_buffer->m_d_ptr;
        }
    }

    void add_pending_op(void* h_buffer, cudaStream_t stream, const std::vector<void*>& d_buffers)
    {
        m_pending_ops.emplace_back(h_buffer, stream, d_buffers);
    }

    void ensure_is_ready(void* h_buffer)
    {
        for (size_t i = 0; i < m_pending_ops.size(); i++)
        {
            auto& op = m_pending_ops[i];

            if (op.m_h_buffer == h_buffer)
            {
                for (void* d_buffer : op.m_d_buffers)
                {
                    bool found = false;
                    for (auto& b : m_d_buffers)
                        if (b.m_d_ptr == d_buffer)
                        {
                            b.m_used = false;
                            found = true;
                            break;
                        }

                    if (!found)
                        throw std::runtime_error("CublasOps::ensure_is_ready: device buffer not found.");
                }

                CUDA_ERRCHECK(cudaStreamSynchronize(op.m_stream));

                m_pending_ops.erase(m_pending_ops.begin() + i);

                return;
            }
        }
    }

    void check_buffers_health()
    {
        if (m_pending_ops.size())
            throw std::runtime_error("CublasOps::check_buffers_health: m_pending_ops not empty.");

        for (auto& b : m_d_buffers)
            if (b.m_used)
                throw std::runtime_error("CublasOps::check_buffers_health: buffer in m_d_buffers still in use.");
    }

    enum class Type
    {
        none,
        uint8,
        fp16,
        fp32
    };

#define CUBLASOPS_TAG_TEXT "OnnxStreamCublas"

    bool is(void* op)
    {
        const char* c0 = (char*)op;
        const char* c1 = CUBLASOPS_TAG_TEXT;
        for (size_t i = 0; i < sizeof(CUBLASOPS_TAG_TEXT); i++)
            if (c0[i] != c1[i])
                return false;
        return true;
    }

    class OpHeader
    {
    public:

        char m_tag[sizeof(CUBLASOPS_TAG_TEXT)];
        CublasOps* m_cublas_ops = nullptr;

        OpHeader(CublasOps* cublas_ops) : m_cublas_ops(cublas_ops)
        {
            strcpy(m_tag, CUBLASOPS_TAG_TEXT);
        }
        virtual ~OpHeader()
        {
        }
    };

    template <typename T>
    T* getptr(void* op)
    {
        if (!is(op))
            throw std::runtime_error("CublasOps::getptr: is(op) == false.");
#define OFFSETOF(TYPE, ELEMENT) ((size_t)&(((TYPE*)0)->ELEMENT)) // c++ offsetof in header <cstddef> gives a warning.
        const size_t offset = OFFSETOF(OpHeader, m_tag);
#undef OFFSETOF
        OpHeader* ptr = (OpHeader*)((unsigned char*)op - offset);
        T* ret = dynamic_cast<T*>(ptr);
        if (!ret)
            throw std::runtime_error("CublasOps::getptr: dynamic_cast failed.");
        return ret;
    }

    void delete_operator(void* op)
    {
        delete getptr<OpHeader>(op);
    }

    class OpFullyConnected : public OpHeader
    {
    public:

        Type m_type = Type::none;
        size_t m_kernel_dim_0 = 0;
        size_t m_kernel_dim_1 = 0;

        void* m_d_kernel = nullptr;

        void run(
            size_t input_dim_0,
            const void* input,
            void* output)
        {
            size_t sizeof_element = m_type == Type::fp16 ? 2 : m_type == Type::fp32 ? 4 : 0;
            size_t sizeof_input = input_dim_0 * m_kernel_dim_0 * sizeof_element;
            size_t sizeof_output = input_dim_0 * m_kernel_dim_1 * sizeof_element;

            cublasHandle_t cublas_handle = m_cublas_ops->get_cublas_handle();
            cudaStream_t stream = m_cublas_ops->get_next_stream();

            CUBLAS_ERRCHECK(cublasSetStream(cublas_handle, stream));

            void* d_input = m_cublas_ops->get_d_buffer(sizeof_input);
            void* d_output = m_cublas_ops->get_d_buffer(sizeof_output);

            CUDA_ERRCHECK(cudaMemcpyAsync(d_input, input, sizeof_input, cudaMemcpyHostToDevice, stream));

            CUDA_ERRCHECK(cudaEventRecord(m_cublas_ops->get_sync_event(), stream));

            float alpha_fp32 = 1.0f;
            float beta_fp32 = 0;
            __half alpha_fp16 = 1.0f;
            __half beta_fp16 = 0;

            cudaDataType data_type = m_type == Type::fp16 ? CUDA_R_16F : CUDA_R_32F;
            cublasComputeType_t compute_type = m_cublas_ops->m_cuda_options.m_compute_fp32 ? CUBLAS_COMPUTE_32F :
                m_type == Type::fp16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;

            CUBLAS_ERRCHECK(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m_kernel_dim_1, input_dim_0, m_kernel_dim_0,
                compute_type == CUBLAS_COMPUTE_16F ? (void*)&alpha_fp16 : (void*)&alpha_fp32,
                m_d_kernel, data_type, m_kernel_dim_1,
                d_input, data_type, m_kernel_dim_0,
                compute_type == CUBLAS_COMPUTE_16F ? (void*)&beta_fp16 : (void*)&beta_fp32,
                d_output, data_type, m_kernel_dim_1,
                compute_type, CUBLAS_GEMM_DEFAULT));

            CUDA_ERRCHECK(cudaMemcpyAsync(output, d_output, sizeof_output, cudaMemcpyDeviceToHost, stream));

            CUDA_ERRCHECK(cudaEventSynchronize(m_cublas_ops->get_sync_event()));

            m_cublas_ops->add_pending_op(output, stream, { d_input, d_output });
        }

        OpFullyConnected(
            CublasOps* cublas_ops,
            Type type,
            size_t kernel_dim_0,
            size_t kernel_dim_1,
            const void* kernel) :
            OpHeader(cublas_ops),
            m_type(type), m_kernel_dim_0(kernel_dim_0), m_kernel_dim_1(kernel_dim_1)
        {
            size_t sizeof_element = type == Type::fp16 ? 2 : type == Type::fp32 ? 4 : 0;
            size_t count = kernel_dim_0 * kernel_dim_1 * sizeof_element;

            CUDA_ERRCHECK(cudaMalloc(&m_d_kernel, count));
            CUDA_ERRCHECK(cudaMemcpy(m_d_kernel, kernel, count, cudaMemcpyHostToDevice));
        }
        virtual ~OpFullyConnected()
        {
            if (m_d_kernel)
            {
                CUDA_ERRCHECK(cudaFree(m_d_kernel));
            }
        }
    };

    void* create_fully_connected(
        Type type,
        size_t input_channels,
        size_t output_channels,
        size_t input_stride,
        size_t output_stride,
        const void* kernel,
        const void* bias)
    {
        if (type != Type::fp16 && type != Type::fp32)
            return nullptr;

        if (input_channels != input_stride || output_channels != output_stride)
            throw std::runtime_error("CublasOps::create_fully_connected: channels != stride.");
        if (bias)
            throw std::runtime_error("CublasOps::create_fully_connected: bias not supported.");

        size_t sizeof_element = type == Type::fp16 ? 2 : type == Type::fp32 ? 4 : 0;
        m_vram_used += input_channels * output_channels * sizeof_element;
        if (m_vram_used > m_cuda_options.m_vram_to_use)
            return nullptr;

        OpFullyConnected* op = new OpFullyConnected(this, type, input_channels, output_channels, kernel);
        return &op->m_tag;
    }

    void run_fully_connected(
        void* fully_connected_op,
        size_t batch_size,
        const void* input,
        void* output)
    {
        getptr<OpFullyConnected>(fully_connected_op)->run(batch_size, input, output);
    }
};

#else

class CublasOps
{
public:

    CudaOptions m_cuda_options;

    enum class Type
    {
        none,
        uint8,
        fp16,
        fp32
    };

    bool is(void* op)
    {
        return false;
    }

    void delete_operator(void* op)
    {
        throw std::runtime_error("Not supported.");
    }

    void ensure_is_ready(void* h_buffer)
    {
        return;
    }

    void check_buffers_health()
    {
        return;
    }

    void* create_fully_connected(
        Type type,
        size_t input_channels,
        size_t output_channels,
        size_t input_stride,
        size_t output_stride,
        const void* kernel,
        const void* bias)
    {
        return nullptr;
    }

    void run_fully_connected(
        void* fully_connected_op,
        size_t batch_size,
        const void* input,
        void* output)
    {
        throw std::runtime_error("Not supported.");
    }
};

#endif

class XnnPack
{
public:

    CublasOps m_cublas_ops;

    pthreadpool_t threadpool = nullptr;

    struct OpsCacheEntry
    {
        xnn_operator_t m_op = nullptr;
    };

    std::unordered_map<std::string, OpsCacheEntry> m_ops_cache;

    void* m_workspace = nullptr;
    size_t m_workspace_size = 0;
    size_t m_workspace_alignment = 0;

public:

    XnnPack(int threads_count)
    {
        threadpool = pthreadpool_create(threads_count);
        if (threadpool == nullptr)
            throw std::runtime_error("failed to create threadpool");

        xnn_status status = xnn_initialize(nullptr /* allocator */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to initialize XNNPACK");
    }

    ~XnnPack() noexcept(false)
    {
        free_ops_cache();

        if (m_workspace)
        {
            custom_aligned_free(m_workspace);
            m_workspace = nullptr;
        }

        if (threadpool)
        {
            pthreadpool_destroy(threadpool);
            threadpool = nullptr;
        }
    }

    void free_ops_cache()
    {
        for (auto& entry : m_ops_cache)
        {
            xnn_operator_t op = entry.second.m_op;

            if (m_cublas_ops.is(op))
            {
                m_cublas_ops.delete_operator(op);
            }
            else
            {
                xnn_status status = xnn_delete_operator(op);
                if (status != xnn_status_success)
                    throw std::runtime_error("failed to delete operator");
            }
        }

        m_ops_cache.clear();
    }

    void parallelize(void* pfn, void* pctx, size_t range)
    {
        pthreadpool_parallelize_1d(threadpool,
            (pthreadpool_task_1d_t)pfn,
            pctx,
            range,
            PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);
    }

    size_t parallelize_threads_count()
    {
        return pthreadpool_get_threads_count(threadpool);
    }

    void allocate_workspace(size_t size, size_t alignment)
    {
        if (size > m_workspace_size || alignment > m_workspace_alignment)
        {
            if (m_workspace)
            {
                custom_aligned_free(m_workspace);
                m_workspace = nullptr;
            }

            m_workspace_size = std::max(m_workspace_size, size);
            m_workspace_alignment = std::max(m_workspace_alignment, alignment);
            m_workspace = custom_aligned_alloc(m_workspace_alignment, m_workspace_size);
        }
    }

    template <typename U, typename T>
    tensor_vector<T> convert(tensor_vector<U>& input)
    {
        const size_t batch_size = input.size();

        tensor_vector<T> output = create_tensor_vector<T>(batch_size);

#if 1

        if (!convert<U, T>(input.data(), output.data(), batch_size, false /* single_threaded */))
            throw std::runtime_error("FP16<->FP32 conversion error.");

#else

        typedef typename std::conditional<std::is_same<T, float>::value, void, float>::type xnn_ptr_type_src;
        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type_dst;

        enum xnn_status(*xnn_create_convert_nc_fxx_fxx)(size_t, size_t, size_t, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_convert_nc_fxx_fxx)(xnn_operator_t, size_t, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_convert_nc_fxx_fxx)(xnn_operator_t, const xnn_ptr_type_src*, xnn_ptr_type_dst*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_convert_nc_fxx_fxx = &xnn_create_convert_nc_f16_f32;
            xnn_reshape_convert_nc_fxx_fxx = &xnn_reshape_convert_nc_f16_f32;
            xnn_setup_convert_nc_fxx_fxx = &xnn_setup_convert_nc_f16_f32;
        }
        else
        {
            xnn_create_convert_nc_fxx_fxx = &xnn_create_convert_nc_f32_f16;
            xnn_reshape_convert_nc_fxx_fxx = &xnn_reshape_convert_nc_f32_f16;
            xnn_setup_convert_nc_fxx_fxx = &xnn_setup_convert_nc_f32_f16;
        }

        xnn_operator_t convert_op = nullptr;
        xnn_status status = xnn_create_convert_nc_fxx_fxx(
            1 /* channels */, 1 /* input stride */, 1 /* output stride */,
            0 /* flags */, &convert_op);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to create F16->F32 Convert operator");

        scope_guard __sg__([&convert_op]() {
            xnn_status status = xnn_delete_operator(convert_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            convert_op = nullptr;
        });

        status = xnn_reshape_convert_nc_fxx_fxx(
            convert_op /* convert_op */,
            batch_size /* batch_size */,
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape Convert operator");

        status = xnn_setup_convert_nc_fxx_fxx(
            convert_op,
            input.data(), output.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup F16->F32 Convert operator");

        status = xnn_run_operator(convert_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run F16->F32 Convert operator");

#endif

        return output;
    }

    template <typename U, typename T>
    bool convert(U* input, T* output, size_t batch_size, bool single_threaded = true)
    {
        typedef typename std::conditional<std::is_same<T, float>::value, void, float>::type xnn_ptr_type_src;
        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type_dst;

        enum xnn_status(*xnn_run_convert_nc_fxx_fxx)(size_t, size_t, size_t, size_t, const xnn_ptr_type_src*, xnn_ptr_type_dst*, uint32_t, pthreadpool_t);

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_run_convert_nc_fxx_fxx = &xnn_run_convert_nc_f16_f32;
        }
        else
        {
            xnn_run_convert_nc_fxx_fxx = &xnn_run_convert_nc_f32_f16;
        }

        xnn_status status = xnn_run_convert_nc_fxx_fxx(
            1 /* channels */,
            1 /* input_stride */,
            1 /* output_stride */,
            batch_size /* batch_size */,
            input /* input */,
            output /* output */,
            0 /* flags */,
            single_threaded ? nullptr : threadpool /* threadpool */);
        if (status != xnn_status_success)
            return false;

        return true;
    }

    template <typename U, typename T>
    bool convert_qu8(U* input, T* output, size_t batch_size, float scale, uint8_t zero_point, bool single_threaded = true)
    {
        typedef typename std::conditional<std::is_same<T, float>::value, uint8_t, float>::type xnn_ptr_type_src;
        typedef typename std::conditional<std::is_same<T, float>::value, float, uint8_t>::type xnn_ptr_type_dst;

        enum xnn_status(*xnn_run_convert_nc_xx_xx)(size_t, size_t, size_t, size_t, const xnn_ptr_type_src*, xnn_ptr_type_dst*, float, uint8_t, uint32_t, pthreadpool_t);

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_run_convert_nc_xx_xx = &xnn_run_convert_nc_qu8_f32;
        }
        else
        {
            xnn_run_convert_nc_xx_xx = &xnn_run_convert_nc_f32_qu8;
        }

        xnn_status status = xnn_run_convert_nc_xx_xx(
            1 /* channels */,
            1 /* input_stride */,
            1 /* output_stride */,
            batch_size /* batch_size */,
            input /* input */,
            output /* output */,
            scale /* input_scale/output_scale */,
            zero_point /* input_zero_point/output_zero_point */,
            0 /* flags */,
            single_threaded ? nullptr : threadpool /* threadpool */);
        if (status != xnn_status_success)
            return false;

        return true;
    }

    struct Qu8MulData
    {
        uint8_t input1_zero_point;
        float input1_scale;
        uint8_t input2_zero_point;
        float input2_scale;
        uint8_t output_zero_point;
        float output_scale;
    };

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> multiply(
        const std::vector<size_t>& first_shape, T* first_data,
        const std::vector<size_t>& second_shape, T* second_data,
        T* output_data_override = nullptr,
        Qu8MulData* qu8_data = nullptr)
    {
        std::vector<size_t> output_shape;

        for (int i = 0; ; i++)
        {
            size_t first = 0, second = 0;

            if (i < first_shape.size())
                first = first_shape[first_shape.size() - 1 - i];
            if (i < second_shape.size())
                second = second_shape[second_shape.size() - 1 - i];

            if (i != 0 && !first && !second)
                break;

            if (!first)
                first = 1;
            if (!second)
                second = 1;

            if (first != 1 && second != 1 && first != second)
                throw std::runtime_error("XnnPack::multiply_fp32: unable to broadcast.");

            output_shape.insert(output_shape.begin(), first == 1 ? second : first);
        }

        size_t output_size = 1;
        for (auto& s : output_shape)
            output_size *= s;

        tensor_vector<T> output_data = create_tensor_vector<T>(output_data_override ? 0 : output_size);

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_multiply_nd_xxx)(float, float, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_multiply_nd_xxx)(xnn_operator_t, size_t, const size_t*, size_t, const size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_multiply_nd_xxx)(xnn_operator_t, const xnn_ptr_type*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_multiply_nd_xxx = &xnn_create_multiply_nd_f32;
            xnn_reshape_multiply_nd_xxx = &xnn_reshape_multiply_nd_f32;
            xnn_setup_multiply_nd_xxx = &xnn_setup_multiply_nd_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_multiply_nd_xxx = &xnn_create_multiply_nd_f16;
            xnn_reshape_multiply_nd_xxx = &xnn_reshape_multiply_nd_f16;
            xnn_setup_multiply_nd_xxx = &xnn_setup_multiply_nd_f16;
        }
        else
        {
            xnn_create_multiply_nd_xxx = nullptr;
            xnn_reshape_multiply_nd_xxx = &xnn_reshape_multiply_nd_qu8;
            xnn_setup_multiply_nd_xxx = &xnn_setup_multiply_nd_qu8;
        }

        xnn_operator_t multiply_op = nullptr;
        xnn_status status;

        if constexpr (!std::is_same<T, uint8_t>::value)
        {
            status = xnn_create_multiply_nd_xxx(
                -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
                0 /* flags */,
                &multiply_op);
        }
        else
        {
            status = xnn_create_multiply_nd_qu8(
                qu8_data->input1_zero_point,
                qu8_data->input1_scale,
                qu8_data->input2_zero_point,
                qu8_data->input2_scale,
                qu8_data->output_zero_point,
                qu8_data->output_scale,
                0 /* output min */, 255 /* output max */,
                0 /* flags */,
                &multiply_op);
        }

        if (status != xnn_status_success)
            throw std::runtime_error("failed to create multiply operation");

        scope_guard __sg__([&multiply_op]() {
            xnn_status status = xnn_delete_operator(multiply_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            multiply_op = nullptr;
        });

        status = xnn_reshape_multiply_nd_xxx(
            multiply_op,
            first_shape.size(), first_shape.data(), second_shape.size(), second_shape.data(),
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape multiply operation");

        status = xnn_setup_multiply_nd_xxx(
            multiply_op,
            first_data /* a */, second_data /* b */, output_data_override ? output_data_override : output_data.data() /* output */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup multiply operation");

        status = xnn_run_operator(multiply_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run multiply operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> matrix_multiply_dynamic(
        const std::vector<size_t>& first_shape, T* first_data,
        const std::vector<size_t>& second_shape, T* second_data,
        std::vector<size_t>* bias_shape, T* bias_data,
        T* output_data_override = nullptr)
    {
        if (first_shape.size() != 2 || second_shape.size() != 2)
            throw std::runtime_error("XnnPack::matrix_multiply_fp32: not implemented (shape of inputs).");

        if (first_shape[1] != second_shape[0])
            throw std::runtime_error("XnnPack::matrix_multiply_fp32: invalid shape of inputs.");

        std::vector<size_t> output_shape({ first_shape[0], second_shape[1] });

        if (bias_data)
        {
            if (bias_shape->size() != output_shape.size() || (*bias_shape)[0] != output_shape[0] || (*bias_shape)[1] != output_shape[1])
                throw std::runtime_error("XnnPack::matrix_multiply_fp32: invalid shape of bias.");
        }

        size_t output_size = output_shape[0] * output_shape[1];
        tensor_vector<T> output_data = create_tensor_vector<T>(output_data_override ? 0 : output_size);

        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type;
        enum xnn_status(*xnn_create_dynamic_fully_connected_nc_xxx)(float, float, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_dynamic_fully_connected_nc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t, size_t, size_t*, size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_dynamic_fully_connected_nc_xxx)(xnn_operator_t, void*, const xnn_ptr_type*, const xnn_ptr_type*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_dynamic_fully_connected_nc_xxx = &xnn_create_dynamic_fully_connected_nc_f32;
            xnn_reshape_dynamic_fully_connected_nc_xxx = &xnn_reshape_dynamic_fully_connected_nc_f32;
            xnn_setup_dynamic_fully_connected_nc_xxx = &xnn_setup_dynamic_fully_connected_nc_f32;
        }
        else
        {
            xnn_create_dynamic_fully_connected_nc_xxx = &xnn_create_dynamic_fully_connected_nc_f16;
            xnn_reshape_dynamic_fully_connected_nc_xxx = &xnn_reshape_dynamic_fully_connected_nc_f16;
            xnn_setup_dynamic_fully_connected_nc_xxx = &xnn_setup_dynamic_fully_connected_nc_f16;
        }

        xnn_operator_t fc_op = nullptr;
        xnn_status status;

        status = xnn_create_dynamic_fully_connected_nc_xxx(
            -std::numeric_limits<float>::infinity() /* output_min */,
            +std::numeric_limits<float>::infinity() /* output_max */,
            XNN_FLAG_TRANSPOSE_WEIGHTS /* flags */,
            &fc_op /* fully_connected_op_out */);

        if (status != xnn_status_success)
            throw std::runtime_error("failed to create fully connected operation");

        scope_guard __sg__([&]() {
            xnn_status status = xnn_delete_operator(fc_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            fc_op = nullptr;
        });

        size_t workspace_size = 0;
        size_t workspace_alignment = 0;

        status = xnn_reshape_dynamic_fully_connected_nc_xxx(
            fc_op,
            first_shape[0] /* batch_size */,
            second_shape[0] /* input_channels */,
            second_shape[1] /* output_channels */,
            second_shape[0] /* input_stride */,
            second_shape[1] /* output_stride */,
            &workspace_size, &workspace_alignment,
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape fully connected operation");

        allocate_workspace(workspace_size, workspace_alignment);

        status = xnn_setup_dynamic_fully_connected_nc_xxx(
            fc_op /* fully_connected_op */,
            m_workspace,
            first_data /* input */,
            second_data /* kernel */,
            !bias_data ? nullptr : bias_data /* bias */,
            output_data_override ? output_data_override : output_data.data() /* output */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup fully connected operation");

        status = xnn_run_operator(fc_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run fully connected operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    };

    struct Qu8MatMulData
    {
        uint8_t input_zero_point;
        float input_scale;
        uint8_t kernel_zero_point;
        float kernel_scale;
        uint8_t output_zero_point;
        float output_scale;
    };

    template <typename T, typename U>
    std::pair<std::vector<size_t>, tensor_vector<T>> matrix_multiply(
        const std::vector<size_t>& first_shape, T* first_data,
        const std::vector<size_t>& second_shape, T* second_data,
        std::vector<size_t>* bias_shape, U* bias_data,
        T* output_data_override = nullptr,
        Qu8MatMulData* qu8_data = nullptr,
        const std::string& cache_key = "")
    {
        if (!cache_key.size())
        {
            if constexpr (std::is_same<T, float>::value)
                return matrix_multiply_dynamic<float>(first_shape, first_data, second_shape, second_data, bias_shape, bias_data, output_data_override);
            else if constexpr (std::is_same<T, uint16_t>::value)
                return matrix_multiply_dynamic<uint16_t>(first_shape, first_data, second_shape, second_data, bias_shape, bias_data, output_data_override);
        }

        if (first_shape.size() != 2 || second_shape.size() != 2)
            throw std::runtime_error("XnnPack::matrix_multiply_fp32: not implemented (shape of inputs).");

        if (first_shape[1] != second_shape[0])
            throw std::runtime_error("XnnPack::matrix_multiply_fp32: invalid shape of inputs.");

        std::vector<size_t> output_shape({ first_shape[0], second_shape[1] });

        if (bias_data)
        {
            if (bias_shape->size() != output_shape.size() || (*bias_shape)[0] != output_shape[0] || (*bias_shape)[1] != output_shape[1])
                throw std::runtime_error("XnnPack::matrix_multiply_fp32: invalid shape of bias.");
        }

        size_t output_size = output_shape[0] * output_shape[1];
        tensor_vector<T> output_data = create_tensor_vector<T>(output_data_override ? 0 : output_size);

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_fully_connected_nc_xxx)(size_t, size_t, size_t, size_t, const xnn_ptr_type*, const xnn_ptr_type*, float, float, uint32_t, xnn_code_cache_t, xnn_weights_cache_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_fully_connected_nc_xxx)(xnn_operator_t, size_t, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_fully_connected_nc_xxx)(xnn_operator_t, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        CublasOps::Type cublas_type = CublasOps::Type::none;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_fully_connected_nc_xxx = &xnn_create_fully_connected_nc_f32;
            xnn_reshape_fully_connected_nc_xxx = &xnn_reshape_fully_connected_nc_f32;
            xnn_setup_fully_connected_nc_xxx = &xnn_setup_fully_connected_nc_f32;
            cublas_type = CublasOps::Type::fp32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_fully_connected_nc_xxx = &xnn_create_fully_connected_nc_f16;
            xnn_reshape_fully_connected_nc_xxx = &xnn_reshape_fully_connected_nc_f16;
            xnn_setup_fully_connected_nc_xxx = &xnn_setup_fully_connected_nc_f16;
            cublas_type = CublasOps::Type::fp16;
        }
        else
        {
            xnn_create_fully_connected_nc_xxx = nullptr;
            xnn_reshape_fully_connected_nc_xxx = &xnn_reshape_fully_connected_nc_qu8;
            xnn_setup_fully_connected_nc_xxx = &xnn_setup_fully_connected_nc_qu8;
            cublas_type = CublasOps::Type::uint8;
        }

        xnn_operator_t fc_op = nullptr;

        if (cache_key.size() &&
            m_ops_cache.count(cache_key))
        {
            fc_op = m_ops_cache[cache_key].m_op;
        }

        xnn_status status;

        if (!fc_op)
        {
            fc_op = !cache_key.size() ? nullptr : (xnn_operator_t)m_cublas_ops.create_fully_connected(
                cublas_type /* type */,
                second_shape[0] /* input_channels */,
                second_shape[1] /* output_channels */,
                second_shape[0] /* input_stride */,
                second_shape[1] /* output_stride */,
                second_data /* kernel */,
                !bias_data ? nullptr : bias_data /* bias */);

            if (!fc_op)
            {
                if constexpr (!std::is_same<T, uint8_t>::value)
                {
                    status = xnn_create_fully_connected_nc_xxx(
                        second_shape[0] /* input_channels */,
                        second_shape[1] /* output_channels */,
                        second_shape[0] /* input_stride */,
                        second_shape[1] /* output_stride */,
                        second_data /* kernel */,
                        !bias_data ? nullptr : bias_data /* bias */,
                        -std::numeric_limits<float>::infinity() /* output_min */,
                        +std::numeric_limits<float>::infinity() /* output_max */,
                        XNN_FLAG_TRANSPOSE_WEIGHTS /* flags */,
                        nullptr, nullptr /* caches */,
                        &fc_op /* fully_connected_op_out */);
                }
                else
                {
                    status = xnn_create_fully_connected_nc_qu8(
                        second_shape[0] /* input_channels */,
                        second_shape[1] /* output_channels */,
                        second_shape[0] /* input_stride */,
                        second_shape[1] /* output_stride */,
                        qu8_data->input_zero_point,
                        qu8_data->input_scale,
                        qu8_data->kernel_zero_point,
                        qu8_data->kernel_scale,
                        second_data /* kernel */,
                        !bias_data ? nullptr : bias_data /* bias */,
                        qu8_data->output_zero_point,
                        qu8_data->output_scale,
                        0 /* output_min */,
                        255 /* output_max */,
                        XNN_FLAG_TRANSPOSE_WEIGHTS /* flags */,
                        nullptr, nullptr /* caches */,
                        &fc_op /* fully_connected_op_out */);
                }

                if (status != xnn_status_success)
                    throw std::runtime_error("failed to create fully connected operation");
            }
        }

        scope_guard __sg__([&]() {
            if (m_cublas_ops.is(fc_op))
                throw std::runtime_error("trying to delete cublas op as an xnn op");
            xnn_status status = xnn_delete_operator(fc_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            fc_op = nullptr;
        });
        
        if (cache_key.size())
        {
            __sg__.m_active = false;

            if (m_ops_cache.count(cache_key) == 0)
                m_ops_cache[cache_key].m_op = fc_op;
        }

        if (m_cublas_ops.is(fc_op))
        {
            m_cublas_ops.run_fully_connected(
                fc_op /* fully_connected_op */,
                first_shape[0] /* batch_size */,
                first_data /* input */,
                output_data_override ? output_data_override : output_data.data() /* output */);
        }
        else
        {
            status = xnn_reshape_fully_connected_nc_xxx(
                fc_op,
                first_shape[0] /* batch_size */,
                threadpool /* threadpool */);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to reshape fully connected operation");

            status = xnn_setup_fully_connected_nc_xxx(
                fc_op /* fully_connected_op */,
                first_data /* input */,
                output_data_override ? output_data_override : output_data.data() /* output */);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to setup fully connected operation");

            status = xnn_run_operator(fc_op, threadpool /* thread pool */);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to run fully connected operator");
        }

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    };

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> sigmoid(
        std::vector<size_t>& input_shape, tensor_vector<T>& input_data)
    {
        std::vector<size_t> output_shape(input_shape);

        size_t output_size = 1;
        for (auto& d : input_shape)
            output_size *= d;

        tensor_vector<T> output_data = create_tensor_vector<T>(output_size);

        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type;
        enum xnn_status(*xnn_create_sigmoid_nc_xxx)(uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_sigmoid_nc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_sigmoid_nc_xxx)(xnn_operator_t, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_sigmoid_nc_xxx = &xnn_create_sigmoid_nc_f32;
            xnn_reshape_sigmoid_nc_xxx = &xnn_reshape_sigmoid_nc_f32;
            xnn_setup_sigmoid_nc_xxx = &xnn_setup_sigmoid_nc_f32;
        }
        else
        {
            xnn_create_sigmoid_nc_xxx = &xnn_create_sigmoid_nc_f16;
            xnn_reshape_sigmoid_nc_xxx = &xnn_reshape_sigmoid_nc_f16;
            xnn_setup_sigmoid_nc_xxx = &xnn_setup_sigmoid_nc_f16;
        }

        xnn_operator_t sigmoid_op = nullptr;
        xnn_status status = xnn_create_sigmoid_nc_xxx(
            0 /* flags */, &sigmoid_op);
        if (status != xnn_status_success || sigmoid_op == nullptr)
            throw std::runtime_error("failed to create Sigmoid operator");

        scope_guard __sg__([&sigmoid_op]() {
            xnn_status status = xnn_delete_operator(sigmoid_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            sigmoid_op = nullptr;
        });

        status = xnn_reshape_sigmoid_nc_xxx(
            sigmoid_op,
            output_size /* batch_size */,
            1 /* channels */, 1 /* input stride */, 1 /* output stride */,
            threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape Sigmoid operator");

        status = xnn_setup_sigmoid_nc_xxx(
            sigmoid_op,
            input_data.data(), output_data.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup Sigmoid operator");

        status = xnn_run_operator(sigmoid_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run Sigmoid operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    struct Qu8ConvData
    {
        uint8_t input_zero_point;
        float input_scale;
        uint8_t kernel_zero_point;
        float kernel_scale;
        uint8_t output_zero_point;
        float output_scale;
    };

    template <typename T, typename U>
    std::pair<std::vector<size_t>, tensor_vector<T>> convolution(
        bool nchw,
        std::vector<size_t>& x_shape, tensor_vector<T>& x_data,
        std::vector<size_t>& w_shape, tensor_vector<T>& w_data,
        U* b_data, size_t b_data_size,
        std::vector<int>& dilations, std::vector<int>& kernel_shape, std::vector<int>& pads, std::vector<int>& strides, int groups,
        Qu8ConvData* qu8_data,
        const std::string& cache_key)
    {
        if (x_shape.size() != 4 || w_shape.size() != 4 ||
            dilations.size() != 2 || dilations[0] != dilations[1] ||
            kernel_shape.size() != 2 || pads.size() != 4 ||
            strides.size() != 2 || strides[0] != strides[1])
        {
            throw std::runtime_error("XnnPack::convolution_nhwc_fp32: one or more arguments are invalid.");
        }

        const size_t batch_size = 1;
        const size_t input_height = nchw ? x_shape[2] : x_shape[1];
        const size_t input_width = nchw ? x_shape[3] : x_shape[2];
        const size_t kernel_height = kernel_shape[0];
        const size_t kernel_width = kernel_shape[1];
        const size_t padding_height = (size_t)(pads[0] + pads[2]);
        const size_t padding_width = (size_t)(pads[1] + pads[3]);
        const size_t subsampling = strides[0];
        const size_t dilation = dilations[0];
        const size_t group_input_channels = nchw ? x_shape[1] : x_shape[3];
        const size_t group_output_channels = w_shape[0];

        const size_t output_pixel_stride = groups * group_output_channels;
        const size_t input_pixel_stride = groups * group_input_channels;
        const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
        const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
        const size_t padding_left = padding_width / 2;
        const size_t padding_top = padding_height / 2;
        const size_t padding_right = padding_width - padding_left;
        const size_t padding_bottom = padding_height - padding_top;
        const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
        const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

        const size_t output_elements = batch_size * output_height * output_width * output_pixel_stride;

        xnn_operator_t convolution_op = nullptr;
        bool from_ops_cache = false;

        if (cache_key.size() &&
            m_ops_cache.count(cache_key))
        {
            convolution_op = m_ops_cache[cache_key].m_op;
            from_ops_cache = true;
        }

        if (x_data.size() != batch_size * input_height * input_width * input_pixel_stride)
            throw std::runtime_error("XnnPack::convolution: invalid size of X.");
        if (!from_ops_cache && w_data.size() != groups * group_output_channels * kernel_height * kernel_width * group_input_channels)
            throw std::runtime_error("XnnPack::convolution: invalid size of W.");
        if (b_data && b_data_size != groups * group_output_channels)
            throw std::runtime_error("XnnPack::convolution: invalid size of B.");

        std::vector<size_t> output_shape({ 1, output_height, output_width, group_output_channels });

        tensor_vector<T> output_data = create_tensor_vector<T>(output_elements);

        enum xnn_status(*xnn_reshape_convolution2d_nchw_f32_)(xnn_operator_t, size_t, size_t, size_t, size_t*, size_t*, size_t*, size_t*, pthreadpool_t) =
            [](
                xnn_operator_t convolution_op,
                size_t batch_size,
                size_t input_height,
                size_t input_width,
                size_t* workspace_size,
                size_t* workspace_alignment,
                size_t* output_height_out,
                size_t* output_width_out,
                pthreadpool_t threadpool)
            {
                return xnn_reshape_convolution2d_nchw_f32(
                    convolution_op,
                    batch_size,
                    input_height,
                    input_width,
                    output_height_out,
                    output_width_out,
                    threadpool);
            };

        enum xnn_status(*xnn_setup_convolution2d_nchw_f32_)(xnn_operator_t, void*, const float*, float*) =
            [](
                xnn_operator_t convolution_op,
                void* workspace,
                const float* input,
                float* output)
            {
                return xnn_setup_convolution2d_nchw_f32(
                    convolution_op,
                    input,
                    output);
            };

        enum xnn_status(*xnn_reshape_convolution2d_nchw_f16_)(xnn_operator_t, size_t, size_t, size_t, size_t*, size_t*, size_t*, size_t*, pthreadpool_t) =
            [](
                xnn_operator_t convolution_op,
                size_t batch_size,
                size_t input_height,
                size_t input_width,
                size_t* workspace_size,
                size_t* workspace_alignment,
                size_t* output_height_out,
                size_t* output_width_out,
                pthreadpool_t threadpool)
            {
                return xnn_reshape_convolution2d_nchw_f16(
                    convolution_op,
                    batch_size,
                    input_height,
                    input_width,
                    output_height_out,
                    output_width_out,
                    threadpool);
            };

        enum xnn_status(*xnn_setup_convolution2d_nchw_f16_)(xnn_operator_t, void*, const void*, void*) =
            [](
                xnn_operator_t convolution_op,
                void* workspace,
                const void* input,
                void* output)
            {
                return xnn_setup_convolution2d_nchw_f16(
                    convolution_op,
                    input,
                    output);
            };

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_convolution2d_nhwc_xxx)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t, size_t, const xnn_ptr_type*, const xnn_ptr_type*, float, float, uint32_t, xnn_code_cache_t, xnn_weights_cache_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_convolution2d_nhwc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t*, size_t*, size_t*, size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_convolution2d_nhwc_xxx)(xnn_operator_t, void*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_convolution2d_nhwc_xxx = nchw ? &xnn_create_convolution2d_nchw_f32 : &xnn_create_convolution2d_nhwc_f32;
            xnn_reshape_convolution2d_nhwc_xxx = nchw ? xnn_reshape_convolution2d_nchw_f32_ : &xnn_reshape_convolution2d_nhwc_f32;
            xnn_setup_convolution2d_nhwc_xxx = nchw ? xnn_setup_convolution2d_nchw_f32_ : &xnn_setup_convolution2d_nhwc_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_convolution2d_nhwc_xxx = nchw ? &xnn_create_convolution2d_nchw_f16 : &xnn_create_convolution2d_nhwc_f16;
            xnn_reshape_convolution2d_nhwc_xxx = nchw ? xnn_reshape_convolution2d_nchw_f16_ : &xnn_reshape_convolution2d_nhwc_f16;
            xnn_setup_convolution2d_nhwc_xxx = nchw ? xnn_setup_convolution2d_nchw_f16_ : &xnn_setup_convolution2d_nhwc_f16;
        }
        else
        {
            if (nchw) throw std::runtime_error("XnnPack::convolution: layout not supported: nchw.");

            xnn_create_convolution2d_nhwc_xxx = nullptr;
            xnn_reshape_convolution2d_nhwc_xxx = &xnn_reshape_convolution2d_nhwc_qu8;
            xnn_setup_convolution2d_nhwc_xxx = &xnn_setup_convolution2d_nhwc_qu8;
        }

        xnn_status status;

        if (!convolution_op)
        {
            if constexpr (!std::is_same<T, uint8_t>::value)
            {
                status = xnn_create_convolution2d_nhwc_xxx(
                    padding_top, padding_right, padding_bottom, padding_left,
                    kernel_height, kernel_width,
                    subsampling, subsampling,
                    dilation, dilation,
                    groups, group_input_channels, group_output_channels,
                    input_pixel_stride, output_pixel_stride,
                    w_data.data(), b_data,
                    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
                    0 /* flags */, nullptr, nullptr, &convolution_op);
            }
            else
            {
                status = xnn_create_convolution2d_nhwc_qu8(
                    padding_top, padding_right, padding_bottom, padding_left,
                    kernel_height, kernel_width,
                    subsampling, subsampling,
                    dilation, dilation,
                    groups, group_input_channels, group_output_channels,
                    input_pixel_stride, output_pixel_stride,
                    qu8_data->input_zero_point, qu8_data->input_scale, qu8_data->kernel_zero_point, qu8_data->kernel_scale,
                    w_data.data(), b_data,
                    qu8_data->output_zero_point, qu8_data->output_scale,
                    0, 255,
                    0 /* flags */, nullptr, nullptr, &convolution_op);
            }

            if (status != xnn_status_success)
                throw std::runtime_error("failed to create FP32 Convolution operator");
        }

        scope_guard __sg__([&convolution_op]() {
            xnn_status status = xnn_delete_operator(convolution_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            convolution_op = nullptr;
        });

        if (cache_key.size())
        {
            __sg__.m_active = false;

            if (m_ops_cache.count(cache_key) == 0)
                m_ops_cache[cache_key].m_op = convolution_op;
        }

        //if (!from_ops_cache) BUG: always reshape the convolution, even in this case!
        {
            size_t workspace_size = 0;
            size_t workspace_alignment = 0;
            status = xnn_reshape_convolution2d_nhwc_xxx(
                convolution_op, batch_size, input_height, input_width,
                &workspace_size, &workspace_alignment,
                /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                /*threadpool=*/ threadpool);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to reshape Convolution operator");
        }

        status = xnn_setup_convolution2d_nhwc_xxx(
            convolution_op,
            nullptr,
            x_data.data(), output_data.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup FP32 Convolution operator");

        status = xnn_run_operator(convolution_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run FP32 Convolution operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> maxpool_nhwc(
        std::vector<size_t>& x_shape, tensor_vector<T>& x_data,
        std::vector<int>& dilations, std::vector<int>& poolings, std::vector<int>& pads, std::vector<int>& strides)
    {
        if (x_shape.size() != 4 ||
            dilations.size() != 2 || dilations[0] != dilations[1] ||
            poolings.size() != 2 || pads.size() != 4 ||
            strides.size() != 2 || strides[0] != strides[1])
        {
            throw std::runtime_error("XnnPack::maxpool_nhwc: one or more arguments are invalid.");
        }

        const size_t batch_size = 1;
        const size_t input_height = x_shape[1];
        const size_t input_width = x_shape[2];
        const size_t channels = x_shape[3];
        const size_t padding_height = (size_t)(pads[0] + pads[2]);
        const size_t padding_width = (size_t)(pads[1] + pads[3]);
        const size_t padding_left = padding_width / 2;
        const size_t padding_top = padding_height / 2;
        const size_t padding_right = padding_width - padding_left;
        const size_t padding_bottom = padding_height - padding_top;
        const size_t pooling_height = poolings[0];
        const size_t pooling_width = poolings[1];
        const size_t stride_height = strides[0];
        const size_t stride_width = strides[1];
        const size_t dilation_height = dilations[0];
        const size_t dilation_width = dilations[1];
        const size_t subsampling = strides[0];
        const size_t dilation = dilations[0];
        const size_t effective_kernel_height = (pooling_height - 1) * dilation + 1;
        const size_t effective_kernel_width = (pooling_width - 1) * dilation + 1;
        const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
        const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

        const size_t output_elements = batch_size * output_height * output_width * channels;

        if (x_data.size() != batch_size * input_height * input_width * channels)
            throw std::runtime_error("XnnPack::maxpool_nhwc: invalid size of X.");

        std::vector<size_t> output_shape({ 1, output_height, output_width, channels });

        tensor_vector<T> output_data = create_tensor_vector<T>(output_elements);

        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type;
        enum xnn_status(*xnn_create_max_pooling2d_nhwc_xxx)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, uint32_t, xnn_operator_t*);
        enum xnn_status(*xnn_reshape_max_pooling2d_nhwc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t*, size_t*, pthreadpool_t);
        enum xnn_status(*xnn_setup_max_pooling2d_nhwc_xxx)(xnn_operator_t, const xnn_ptr_type*, xnn_ptr_type*);

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_max_pooling2d_nhwc_xxx = &xnn_create_max_pooling2d_nhwc_f32;
            xnn_reshape_max_pooling2d_nhwc_xxx = &xnn_reshape_max_pooling2d_nhwc_f32;
            xnn_setup_max_pooling2d_nhwc_xxx = &xnn_setup_max_pooling2d_nhwc_f32;
        }
        else
        {
            xnn_create_max_pooling2d_nhwc_xxx = &xnn_create_max_pooling2d_nhwc_f16;
            xnn_reshape_max_pooling2d_nhwc_xxx = &xnn_reshape_max_pooling2d_nhwc_f16;
            xnn_setup_max_pooling2d_nhwc_xxx = &xnn_setup_max_pooling2d_nhwc_f16;
        }

        xnn_operator_t maxpool_op = nullptr;
        xnn_status status;

        status = xnn_create_max_pooling2d_nhwc_xxx(
            padding_top,
            padding_right,
            padding_bottom,
            padding_left,
            pooling_height,
            pooling_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            -std::numeric_limits<float>::infinity(),
            +std::numeric_limits<float>::infinity(),
            /*flags*/ 0,
            &maxpool_op);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to create maxpool operator");

        scope_guard __sg__([&maxpool_op]() {
            xnn_status status = xnn_delete_operator(maxpool_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            maxpool_op = nullptr;
            });

        status = xnn_reshape_max_pooling2d_nhwc_xxx(
            maxpool_op,
            batch_size,
            input_height,
            input_width,
            channels,
            /*input_pixel_stride*/ channels,
            /*output_pixel_stride*/ channels,
            /*output_height_out*/ nullptr,
            /*output_width_out*/ nullptr,
            threadpool);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape maxpool operator");

        status = xnn_setup_max_pooling2d_nhwc_xxx(
            maxpool_op,
            x_data.data(),
            output_data.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup maxpool operator");

        status = xnn_run_operator(maxpool_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run maxpool operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    struct Qu8AddData
    {
        uint8_t input1_zero_point;
        float input1_scale;
        uint8_t input2_zero_point;
        float input2_scale;
        uint8_t output_zero_point;
        float output_scale;
    };

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> add(
        std::vector<size_t>& first_shape, tensor_vector<T>& first_data,
        std::vector<size_t>& second_shape, tensor_vector<T>& second_data,
        Qu8AddData* qu8_data = nullptr)
    {
        std::vector<size_t> output_shape;

        for (int i = 0; ; i++)
        {
            size_t first = 0, second = 0;

            if (i < first_shape.size())
                first = first_shape[first_shape.size() - 1 - i];
            if (i < second_shape.size())
                second = second_shape[second_shape.size() - 1 - i];

            if (i != 0 && !first && !second)
                break;

            if (!first)
                first = 1;
            if (!second)
                second = 1;

            if (first != 1 && second != 1 && first != second)
                throw std::runtime_error("XnnPack::add_fp32: unable to broadcast.");

            output_shape.insert(output_shape.begin(), first == 1 ? second : first);
        }

        size_t output_size = 1;
        for (auto& s : output_shape)
            output_size *= s;

        tensor_vector<T> output_data = create_tensor_vector<T>(output_size);

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_add_nd_xxx)(float, float, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_add_nd_xxx)(xnn_operator_t, size_t, const size_t*, size_t, const size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_add_nd_xxx)(xnn_operator_t, const xnn_ptr_type*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_add_nd_xxx = &xnn_create_add_nd_f32;
            xnn_reshape_add_nd_xxx = &xnn_reshape_add_nd_f32;
            xnn_setup_add_nd_xxx = &xnn_setup_add_nd_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_add_nd_xxx = &xnn_create_add_nd_f16;
            xnn_reshape_add_nd_xxx = &xnn_reshape_add_nd_f16;
            xnn_setup_add_nd_xxx = &xnn_setup_add_nd_f16;
        }
        else
        {
            xnn_create_add_nd_xxx = nullptr;
            xnn_reshape_add_nd_xxx = &xnn_reshape_add_nd_qu8;
            xnn_setup_add_nd_xxx = &xnn_setup_add_nd_qu8;
        }

        xnn_operator_t add_op = nullptr;
        xnn_status status;

        if constexpr (!std::is_same<T, uint8_t>::value)
        {
            status = xnn_create_add_nd_xxx(
                -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
                0 /* flags */,
                &add_op);
        }
        else
        {
            status = xnn_create_add_nd_qu8(
                qu8_data->input1_zero_point,
                qu8_data->input1_scale,
                qu8_data->input2_zero_point,
                qu8_data->input2_scale,
                qu8_data->output_zero_point,
                qu8_data->output_scale,
                0 /* output min */, 255 /* output max */,
                0 /* flags */,
                &add_op);
        }

        if (status != xnn_status_success)
            throw std::runtime_error("failed to create add operation");

        scope_guard __sg__([&add_op]() {
            xnn_status status = xnn_delete_operator(add_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            add_op = nullptr;
        });

        status = xnn_reshape_add_nd_xxx(
            add_op,
            first_shape.size(), first_shape.data(), second_shape.size(), second_shape.data(),
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape add operation");

        status = xnn_setup_add_nd_xxx(
            add_op,
            first_data.data() /* a */, second_data.data() /* b */, output_data.data() /* output */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup add operation");

        status = xnn_run_operator(add_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run add operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> transpose(
        std::vector<size_t>& input_shape, tensor_vector<T>& input_data,
        const std::vector<size_t>& perm)
    {
        if (input_shape.size() == 0 || input_data.size() == 0 || perm.size() == 0 || input_shape.size() != perm.size())
            throw std::runtime_error("XnnPack::transpose: one or more invalid arguments.");

        std::vector<size_t> output_shape;

        for (auto& d : perm)
        {
            if (d >= input_shape.size())
                throw std::runtime_error("XnnPack::transpose: invalid index in perm.");

            output_shape.push_back(input_shape[d]);
        }

        size_t el_count_input = 1;
        for (auto& d : input_shape)
            el_count_input *= d;

        size_t el_count_output = 1;
        for (auto& d : output_shape)
            el_count_output *= d;

        if (el_count_input != el_count_output)
            throw std::runtime_error("XnnPack::transpose: invalid element count in shape of output.");

        tensor_vector<T> output_data = create_tensor_vector<T>(el_count_output);

        enum xnn_status(*xnn_create_transpose_nd_xxx)(uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_transpose_nd_xxx)(xnn_operator_t, size_t, const size_t*, const size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_transpose_nd_xxx)(xnn_operator_t, const void*, void*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_transpose_nd_xxx = &xnn_create_transpose_nd_x32;
            xnn_reshape_transpose_nd_xxx = &xnn_reshape_transpose_nd_x32;
            xnn_setup_transpose_nd_xxx = &xnn_setup_transpose_nd_x32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_transpose_nd_xxx = &xnn_create_transpose_nd_x16;
            xnn_reshape_transpose_nd_xxx = &xnn_reshape_transpose_nd_x16;
            xnn_setup_transpose_nd_xxx = &xnn_setup_transpose_nd_x16;
        }
        else if constexpr (std::is_same<T, uint8_t>::value)
        {
            xnn_create_transpose_nd_xxx = &xnn_create_transpose_nd_x8;
            xnn_reshape_transpose_nd_xxx = &xnn_reshape_transpose_nd_x8;
            xnn_setup_transpose_nd_xxx = &xnn_setup_transpose_nd_x8;
        }
        else
            throw std::runtime_error("XnnPack::transpose: invalid type.");

        xnn_operator_t transpose_op = nullptr;
        xnn_status status = xnn_create_transpose_nd_xxx(
            0 /* flags */,
            &transpose_op);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to create transpose operation");

        scope_guard __sg__([&transpose_op]() {
            xnn_status status = xnn_delete_operator(transpose_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            transpose_op = nullptr;
        });

        status = xnn_reshape_transpose_nd_xxx(
            transpose_op,
            perm.size() /* num_dims */,
            input_shape.data() /* shape */,
            perm.data() /* perm */,
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape transpose operation");

        status = xnn_setup_transpose_nd_xxx(
            transpose_op,
            input_data.data() /* input */,
            output_data.data() /* output */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup transpose operation");

        status = xnn_run_operator(transpose_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run transpose operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> subtract(
        std::vector<size_t>& first_shape, tensor_vector<T>& first_data,
        std::vector<size_t>& second_shape, tensor_vector<T>& second_data)
    {
        std::vector<size_t> output_shape;

        for (int i = 0; ; i++)
        {
            size_t first = 0, second = 0;

            if (i < first_shape.size())
                first = first_shape[first_shape.size() - 1 - i];
            if (i < second_shape.size())
                second = second_shape[second_shape.size() - 1 - i];

            if (i != 0 && !first && !second)
                break;

            if (!first)
                first = 1;
            if (!second)
                second = 1;

            if (first != 1 && second != 1 && first != second)
                throw std::runtime_error("XnnPack::subtract_fp32: unable to broadcast.");

            output_shape.insert(output_shape.begin(), first == 1 ? second : first);
        }

        size_t output_size = 1;
        for (auto& s : output_shape)
            output_size *= s;

        tensor_vector<T> output_data = create_tensor_vector<T>(output_size);

        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type;
        enum xnn_status(*xnn_create_subtract_nd_xxx)(float, float, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_subtract_nd_xxx)(xnn_operator_t, size_t, const size_t*, size_t, const size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_subtract_nd_xxx)(xnn_operator_t, const xnn_ptr_type*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_subtract_nd_xxx = &xnn_create_subtract_nd_f32;
            xnn_reshape_subtract_nd_xxx = &xnn_reshape_subtract_nd_f32;
            xnn_setup_subtract_nd_xxx = &xnn_setup_subtract_nd_f32;
        }
        else
        {
            xnn_create_subtract_nd_xxx = &xnn_create_subtract_nd_f16;
            xnn_reshape_subtract_nd_xxx = &xnn_reshape_subtract_nd_f16;
            xnn_setup_subtract_nd_xxx = &xnn_setup_subtract_nd_f16;
        }

        xnn_operator_t subtract_op = nullptr;
        xnn_status status = xnn_create_subtract_nd_xxx(
            -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
            0 /* flags */,
            &subtract_op);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to create subtract operation");

        scope_guard __sg__([&subtract_op]() {
            xnn_status status = xnn_delete_operator(subtract_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            subtract_op = nullptr;
        });

        status = xnn_reshape_subtract_nd_xxx(
            subtract_op,
            first_shape.size(), first_shape.data(), second_shape.size(), second_shape.data(),
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape subtract operation");

        status = xnn_setup_subtract_nd_xxx(
            subtract_op,
            first_data.data() /* a */, second_data.data() /* b */, output_data.data() /* output */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup subtract operation");

        status = xnn_run_operator(subtract_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run subtract operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> divide(
        std::vector<size_t>& first_shape, tensor_vector<T>& first_data,
        std::vector<size_t>& second_shape, tensor_vector<T>& second_data)
    {
        std::vector<size_t> output_shape;

        for (int i = 0; ; i++)
        {
            size_t first = 0, second = 0;

            if (i < first_shape.size())
                first = first_shape[first_shape.size() - 1 - i];
            if (i < second_shape.size())
                second = second_shape[second_shape.size() - 1 - i];

            if (i != 0 && !first && !second)
                break;

            if (!first)
                first = 1;
            if (!second)
                second = 1;

            if (first != 1 && second != 1 && first != second)
                throw std::runtime_error("XnnPack::divide_fp32: unable to broadcast.");

            output_shape.insert(output_shape.begin(), first == 1 ? second : first);
        }

        size_t output_size = 1;
        for (auto& s : output_shape)
            output_size *= s;

        tensor_vector<T> output_data = create_tensor_vector<T>(output_size);

        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type;
        enum xnn_status(*xnn_create_divide_nd_xxx)(float, float, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_divide_nd_xxx)(xnn_operator_t, size_t, const size_t*, size_t, const size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_divide_nd_xxx)(xnn_operator_t, const xnn_ptr_type*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_divide_nd_xxx = &xnn_create_divide_nd_f32;
            xnn_reshape_divide_nd_xxx = &xnn_reshape_divide_nd_f32;
            xnn_setup_divide_nd_xxx = &xnn_setup_divide_nd_f32;
        }
        else
        {
            xnn_create_divide_nd_xxx = &xnn_create_divide_nd_f16;
            xnn_reshape_divide_nd_xxx = &xnn_reshape_divide_nd_f16;
            xnn_setup_divide_nd_xxx = &xnn_setup_divide_nd_f16;
        }

        xnn_operator_t divide_op = nullptr;
        xnn_status status = xnn_create_divide_nd_xxx(
            -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
            0 /* flags */,
            &divide_op);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to create divide operation");

        scope_guard __sg__([&divide_op]() {
            xnn_status status = xnn_delete_operator(divide_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            divide_op = nullptr;
        });

        status = xnn_reshape_divide_nd_xxx(
            divide_op,
            first_shape.size(), first_shape.data(), second_shape.size(), second_shape.data(),
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape divide operation");

        status = xnn_setup_divide_nd_xxx(
            divide_op,
            first_data.data() /* a */, second_data.data() /* b */, output_data.data() /* output */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup divide operation");

        status = xnn_run_operator(divide_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run divide operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    struct Qu8SoftmaxData
    {
        float input_scale;
        uint8_t output_zero_point;
        float output_scale;
    };

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> softmax(
        std::vector<size_t>& input_shape, T* input_data,
        T* output_data_override = nullptr,
        Qu8SoftmaxData* qu8_data = nullptr,
        size_t channels_override = 0)
    {
        std::vector<size_t> output_shape(input_shape);

        size_t output_size = 1;
        for (auto& d : input_shape)
            output_size *= d;

        tensor_vector<T> output_data = create_tensor_vector<T>(output_data_override ? 0 : output_size);

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_softmax_nc_xxx)(uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_softmax_nc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_softmax_nc_xxx)(xnn_operator_t, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_softmax_nc_xxx = &xnn_create_softmax_nc_f32;
            xnn_reshape_softmax_nc_xxx = &xnn_reshape_softmax_nc_f32;
            xnn_setup_softmax_nc_xxx = &xnn_setup_softmax_nc_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_softmax_nc_xxx = &xnn_create_softmax_nc_f16;
            xnn_reshape_softmax_nc_xxx = &xnn_reshape_softmax_nc_f16;
            xnn_setup_softmax_nc_xxx = &xnn_setup_softmax_nc_f16;
        }
        else
        {
            xnn_create_softmax_nc_xxx = nullptr;
            xnn_reshape_softmax_nc_xxx = &xnn_reshape_softmax_nc_qu8;
            xnn_setup_softmax_nc_xxx = &xnn_setup_softmax_nc_qu8;
        }

        size_t channels = channels_override ? channels_override : input_shape.back();
        size_t batch_size = output_size / channels;

        xnn_operator_t softmax_op = nullptr;
        xnn_status status;

        if constexpr (!std::is_same<T, uint8_t>::value)
        {
            status = xnn_create_softmax_nc_xxx(
                0 /* flags */, &softmax_op);
        }
        else
        {
            status = xnn_create_softmax_nc_qu8(
                qu8_data->input_scale,
                qu8_data->output_zero_point,
                qu8_data->output_scale,
                0 /* flags */, &softmax_op);
        }

        if (status != xnn_status_success || softmax_op == nullptr)
            throw std::runtime_error("failed to create softmax operator");

        scope_guard __sg__([&softmax_op]() {
            xnn_status status = xnn_delete_operator(softmax_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            softmax_op = nullptr;
        });

        status = xnn_reshape_softmax_nc_xxx(
            softmax_op,
            channels /* channels */, channels /* input stride */, channels /* output stride */,
            batch_size /* batch_size */,
            threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape softmax operator");

        status = xnn_setup_softmax_nc_xxx(
            softmax_op,
            input_data, output_data_override ? output_data_override : output_data.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup softmax operator");

        status = xnn_run_operator(softmax_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run softmax operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
    }

    template <typename T>
    std::pair<std::vector<size_t>, tensor_vector<T>> scaled_dot_product_attention(
        tensor_vector<T>& query, tensor_vector<T>& key, tensor_vector<T>& value,
        tensor_vector<T>& scale, tensor_vector<T>& mask,
        size_t batch_size, size_t query_heads, size_t query_tokens,
        size_t key_value_heads, size_t key_value_tokens,
        size_t query_key_channels, size_t value_channels)
    {
        if (query.size() != batch_size * query_heads * query_tokens * query_key_channels)
            throw std::runtime_error("XnnPack::scaled_dot_product_attention: invalid size of query.");
        if (key.size() != batch_size * key_value_heads * key_value_tokens * query_key_channels)
            throw std::runtime_error("XnnPack::scaled_dot_product_attention: invalid size of key.");
        if (value.size() != batch_size * key_value_heads * key_value_tokens * value_channels)
            throw std::runtime_error("XnnPack::scaled_dot_product_attention: invalid size of value.");
        if (scale.size() != query_key_channels)
            throw std::runtime_error("XnnPack::scaled_dot_product_attention: invalid size of scale.");
        if (mask.size() != query_tokens * key_value_tokens)
            throw std::runtime_error("XnnPack::scaled_dot_product_attention: invalid size of mask.");

        std::vector<size_t> output_shape({ batch_size , query_heads , query_tokens , value_channels });

        size_t output_size = 1;
        for (auto& d : output_shape)
            output_size *= d;

        tensor_vector<T> output = create_tensor_vector<T>(output_size);

        typedef typename std::conditional<std::is_same<T, float>::value, float, void>::type xnn_ptr_type;
        enum xnn_status(*xnn_create_scaled_dot_product_attention_nhtc_xxx)(enum xnn_attention_logits_cap_type, const void*, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_scaled_dot_product_attention_nhtc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t*, size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_scaled_dot_product_attention_nhtc_xxx)(xnn_operator_t, void*, const xnn_ptr_type*, const xnn_ptr_type*, const xnn_ptr_type*, const xnn_ptr_type*, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        std::string cache_key;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_scaled_dot_product_attention_nhtc_xxx = &xnn_create_scaled_dot_product_attention_nhtc_f32;
            xnn_reshape_scaled_dot_product_attention_nhtc_xxx = &xnn_reshape_scaled_dot_product_attention_nhtc_f32;
            xnn_setup_scaled_dot_product_attention_nhtc_xxx = &xnn_setup_scaled_dot_product_attention_nhtc_f32;

            cache_key = ":xnn_create_scaled_dot_product_attention_nhtc_f32";
        }
        else
        {
            xnn_create_scaled_dot_product_attention_nhtc_xxx = &xnn_create_scaled_dot_product_attention_nhtc_f16;
            xnn_reshape_scaled_dot_product_attention_nhtc_xxx = &xnn_reshape_scaled_dot_product_attention_nhtc_f16;
            xnn_setup_scaled_dot_product_attention_nhtc_xxx = &xnn_setup_scaled_dot_product_attention_nhtc_f16;

            cache_key = ":xnn_create_scaled_dot_product_attention_nhtc_f16";
        }

        xnn_status status;

        xnn_operator_t attention_op = m_ops_cache.count(cache_key) ? m_ops_cache[cache_key].m_op : nullptr;

        if (!attention_op)
        {
            status = xnn_create_scaled_dot_product_attention_nhtc_xxx(
                xnn_attention_logits_cap_type_none,
                nullptr,
                /*flags=*/0,
                &attention_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to create scaled dot product attention operator");

            m_ops_cache[cache_key].m_op = attention_op;
        }

        size_t workspace_size = 0;
        size_t workspace_alignment = 0;

        status = xnn_reshape_scaled_dot_product_attention_nhtc_xxx(
            attention_op,
            batch_size, query_heads, query_tokens,
            key_value_heads, key_value_tokens,
            query_key_channels, value_channels,
            &workspace_size, &workspace_alignment,
            threadpool);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape scaled dot product attention operator");

        allocate_workspace(workspace_size, workspace_alignment);

        status = xnn_setup_scaled_dot_product_attention_nhtc_xxx(
            attention_op,
            m_workspace, query.data(), key.data(), value.data(),
            scale.data(), mask.data(), output.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup scaled dot product attention operator");

        status = xnn_run_operator(attention_op, threadpool);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run scaled dot product attention operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output));
    }
};

// ---

std::string next_file_line(std::vector<char>& file, size_t& pos)
{
    for (; pos < file.size(); pos++)
        if (file[pos] != '\r' && file[pos] != '\n')
            break;

    auto start = pos;

    for (; pos < file.size(); pos++)
        if (file[pos] == '\r' || file[pos] == '\n')
            break;

    if (pos == start)
        return "";

    return std::string(&file[start], pos - start);
}

std::vector<std::string> split_string(std::string& str, const std::string& delimiter)
{
    std::vector<std::string> ret;

    size_t pos = 0;

    while ((pos = str.find(delimiter)) != std::string::npos) {
        std::string token = str.substr(0, pos);
        ret.push_back(std::move(token));
        str.erase(0, pos + delimiter.length());
    }

    ret.push_back(std::move(str));

    return ret;
}

std::string& ltrim(std::string& s)
{
    s.erase(0, s.find_first_not_of(" \t\n\r\f\v"));
    return s;
}

std::string& rtrim(std::string& s)
{
    s.erase(s.find_last_not_of(" \t\n\r\f\v") + 1);
    return s;
}

std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

template<typename T>
std::vector<T> string_to_int_vec(std::string s)
{
    std::vector<T> ret;

    for (auto& token : split_string(s, ","))
        ret.push_back(std::stoi(token));

    return ret;
}

template<class L, class I>
bool are_all_equal(const L& l, const std::initializer_list<I>& i)
{
    return std::equal(std::begin(l), std::end(l), std::begin(i), std::end(i));
}

class FloatAsUInt
{
public:

    static float uint32_to_f32(uint32_t n)
    {
        return *(float*)&n;
    }

    static float f16_to_f32(uint16_t n)
    {
        uint32_t ret = 0;

        const uint16_t exp_mask = 0b11111 << 10;
        const uint16_t mant_mask = 0b1111111111;

        uint32_t sign = n >> 15;
        uint32_t exp = (n & exp_mask) >> 10;
        uint32_t mant = n & mant_mask;

        if (!exp)
        {
            if (!mant) // +0 or -0
            {
                ret = (sign << 31);
            }
            else // subnormal f16 -> normal f32
            {
                exp = 127 - 14;
                while (!(mant & (1 << 10)))
                {
                    exp--;
                    mant <<= 1;
                }
                mant &= mant_mask;

                ret = (sign << 31) | (exp << 23) | (mant << 23 - 10);
            }
        }
        else if ((n & exp_mask) == exp_mask) // inf or NaN
        {
            ret = (sign << 31) | (0xFF << 23) | (mant << 23 - 10);
        }
        else // normal
        {
            ret = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 23 - 10);
        }

        return *(float*)&ret;
    }

    template<typename T, typename P>
    static T* binary_search_first_eq(T* table, int num, const P& pred)
    {
        T* ret_val = nullptr;

        int start_index = 0;
        int end_index = num - 1;

        while (start_index <= end_index)
        {
            int middle = start_index + (end_index - start_index) / 2;

            auto& s = table[middle];

            auto comp = pred(s);
            if (comp > 0)
                start_index = middle + 1;
            else
            {
                if (!comp) ret_val = &s;
                end_index = middle - 1;
            }
        }

        return ret_val;
    }

    template<typename T>
    static std::optional<std::pair<T, T>> get_percentiles(T* begin, T* end, size_t from_left, size_t from_right)
    {
        T sign_bit, exp_mask;

        if constexpr (std::is_same<T, uint16_t>::value)
        {
            sign_bit = 1 << 15;
            exp_mask = 0b11111 << 10;
        }
        else if constexpr (std::is_same<T, uint32_t>::value)
        {
            sign_bit = 1 << 31;
            exp_mask = 0b11111111 << 23;
        }
        else
            throw std::invalid_argument("get_percentiles: invalid type.");

        T* result_left = nullptr;
        T* result_right = nullptr;

        std::sort(begin, end); // result example: 1.0 2.0 3.0 inf -0.0 -2.0 -inf -nan

        T sign_exp_mask = sign_bit | exp_mask;

        {
            T* first_neg_inf = binary_search_first_eq(begin, end - begin,
                [&](const auto& v) { return (v & sign_exp_mask) == sign_exp_mask ? 0 : v < sign_exp_mask ? +1 : -1; });

            if (first_neg_inf) end = first_neg_inf;
        }

        T* first_neg = binary_search_first_eq(begin, end - begin,
            [&](const auto& v) { return v & sign_bit ? 0 : v < sign_bit ? +1 : -1; });

        T* first_pos_inf = binary_search_first_eq(begin, (first_neg ? first_neg : end) - begin,
            [&](const auto& v) { return (v & exp_mask) == exp_mask ? 0 : v < exp_mask ? +1 : -1; });

        T* end_pos = first_pos_inf ? first_pos_inf : first_neg ? first_neg : end;

        if (first_neg)
        {
            size_t num_negs = end - first_neg;

            if (from_left < num_negs)
            {
                result_left = end - 1 - from_left;
            }
            else
            {
                result_left = &begin[from_left - num_negs];
                if (result_left >= end_pos) result_left = nullptr;
            }
        }
        else
        {
            result_left = &begin[from_left];
            if (result_left >= end_pos) result_left = nullptr;
        }

        if (end_pos > begin)
        {
            size_t num_poss = end_pos - begin;

            if (from_right < num_poss)
            {
                result_right = end_pos - 1 - from_right;
            }
            else if (first_neg)
            {
                result_right = &first_neg[from_right - num_poss];
                if (result_right >= end) result_right = nullptr;
            }
        }
        else if (first_neg)
        {
            result_right = &first_neg[from_right];
            if (result_right >= end) result_right = nullptr;
        }

        if (!result_left || !result_right)
            return std::nullopt;
        else
            return std::make_pair(*result_left, *result_right);
    }
};

// ---

const size_t Model::m_perthread_buffer_size = 64 * 1024;
const size_t Model::m_float16_buffer_size = m_perthread_buffer_size / sizeof(uint16_t);
const size_t Model::m_float32_buffer_size = m_perthread_buffer_size / sizeof(float);
const size_t Model::m_float32_buffer_size_w_extra_bytes = m_float32_buffer_size + XNN_EXTRA_BYTES / sizeof(float);

Model::Model(int threads_count /*= 0*/)
{
    if (threads_count >= 0)
        m_xnnpack = new XnnPack(threads_count);
}

Model::~Model()
{
    if (m_xnnpack)
        delete m_xnnpack;
}

void Model::set_cuda_options(const CudaOptions& options)
{
    m_xnnpack->m_cublas_ops.m_cuda_options = options;
}

void Model::read_file(const char* filename)
{
    auto model = onnxstream::read_file<std::vector<char>>(filename);

    m_model = std::move(model);

    m_path = "";

    auto path_sep = std::string(filename).find_last_of("/\\");
    if (path_sep != std::string::npos)
        m_path = std::string(filename).substr(0, path_sep + 1);

    get_wp()->m_path = m_path;
}

void Model::read_string(const char* string, const char* path_with_slash /*= "./"*/)
{
    m_model.clear();

    size_t len = std::strlen(string);
    m_model.reserve(len);
    for (size_t i = 0; i < len; i++)
        m_model.push_back(string[i]);

    m_path = path_with_slash;
    get_wp()->m_path = m_path;
}

std::string Model::next_line()
{
    return next_file_line(m_model, m_pos);
}

std::optional<Operation> Model::next_op_impl()
{
    auto line = next_line();

    if (!line.size())
        return std::nullopt;

    auto vec = split_string(line, "*");

    if (vec.size() != 3 && vec.size() != 4)
        throw std::invalid_argument("Model::next_op: invalid format of model line.");

    Operation op;

    auto first = split_string(vec[0], ":");

    if (first.size() != 2)
        throw std::invalid_argument("Model::next_op: invalid format of model line.");

    op.m_name = std::move(first[0]);
    op.m_type = std::move(first[1]);

    if (!op.m_name.size())
        op.m_name = "onnxstream_fallback_name_" + std::to_string(m_pos);

    if (vec[1].find("input:") != 0)
        throw std::invalid_argument("Model::next_op: invalid format of model line.");
    else
        vec[1].erase(0, 6);

    auto second = split_string(vec[1], ";");

    for (auto& str : second)
    {
        auto t = parse_tensor_string(str);
        op.m_input.push_back(std::move(t));
    }

    if (vec[2].find("output:") != 0)
        throw std::invalid_argument("Model::next_op: invalid format of model line.");
    else
        vec[2].erase(0, 7);

    auto third = split_string(vec[2], ";");

    for (auto& str : third)
    {
        auto t = parse_tensor_string(str);
        op.m_output.push_back(std::move(t));
    }

    if (vec.size() == 4)
    {
        auto fourth = split_string(vec[3], ";");

        for (auto& str_pair : fourth)
        {
            auto pair = split_string(str_pair, ":");
            if (pair.size() != 2)
                throw std::invalid_argument("Model::next_op: invalid format of model line.");

            op.m_attributes.emplace_back(std::move(pair[0]), std::move(pair[1]));
        }
    }

    return op;
}

std::optional<Operation> Model::next_op()
{
    if (!m_use_next_op_cache)
    {
        return next_op_impl();
    }
    else
    {
        if (!m_next_op_cache_ready)
        {
            std::optional<Operation> op = next_op_impl();
            if (op)
                m_next_op_cache.push_back(*op);
            else
                m_next_op_cache_ready = true;
            return op;
        }
        else
        {
            if (m_pos < m_next_op_cache.size())
                return m_next_op_cache[m_pos++];
            else
                return std::nullopt;
        }
    }
}

Tensor Model::parse_tensor_string(std::string& str)
{
    if (str.size() == 0)
        return Tensor();

    Tensor t;

    auto vec = split_string(str, "(");

    if (vec.size() != 2)
        throw std::invalid_argument("Model::parse_tensor_string: invalid tensor format.");

    if (vec[0].size() == 0 || vec[1].size() == 0 || vec[1].find(')') != vec[1].size() - 1)
        throw std::invalid_argument("Model::parse_tensor_string: invalid tensor format.");
    else
        vec[1].erase(vec[1].size() - 1, 1);

    t.m_name = std::move(vec[0]);

    auto type_vec = split_string(vec[1], ":");

    std::string shape;

    if (type_vec.size() == 1)
    {
        shape = std::move(type_vec[0]);
    }
    else if (type_vec.size() == 2)
    {
        shape = std::move(type_vec[1]);

        if (type_vec[0].find("uint8[") == 0 && type_vec[0].back() == ']')
        {
            auto range = type_vec[0].substr(6, type_vec[0].size() - 6 - 1);
            auto range_vec = split_string(range, ",");
            if (range_vec.size() != 2)
                throw std::invalid_argument("Model::parse_tensor_string: invalid uint8 range.");
            t.m_type = TensorDataType::uint8;
            t.m_scale = std::stod(range_vec[0]);
            t.m_zero_point = std::stoi(range_vec[1]);
        }
        else if (type_vec[0] == "float16")
        {
            t.m_type = TensorDataType::float16;
        }
        else if (type_vec[0] == "float32")
        {
            t.m_type = TensorDataType::float32;
        }
        else if (type_vec[0] == "int64")
        {
            t.m_type = TensorDataType::int64;
        }
        else
            throw std::invalid_argument("Model::parse_tensor_string: unsupported tensor data format.");
    }
    else
        throw std::invalid_argument("Model::parse_tensor_string: invalid tensor format.");

    if (shape.size() != 0)
    {
        auto dims = split_string(shape, ",");

        for (auto& dim : dims)
        {
            int i = std::stoi(dim);
            if (i < 0)
                throw std::invalid_argument("Model::parse_tensor_string: invalid shape (dim < 0).");
            else if (i == 0 && !m_support_dynamic_shapes)
                throw std::invalid_argument("Model::parse_tensor_string: invalid shape (dim == 0).");

            t.m_shape.push_back(i);
        }
    }

    return t;
}

Tensor& Model::get_tensor_data(Tensor& t, bool make_copy /*= false*/, bool requires_float /*= false*/, TensorDataLayout required_layout /*= TensorDataLayout::unspecified*/)
{
    bool load = true;
    bool unique = false;

    if (m_batch_size > 1 && m_batch_index != 0)
    {
        if (m_batch_index >= m_batch_size)
            throw std::invalid_argument("Model::get_tensor_data: invalid m_batch_index.");

        BatchCacheItem* item = nullptr;
        for (auto& c : m_batch_cache)
            if (c.m_index == m_batch_index - 1)
            {
                item = &c;
                item->m_index = m_batch_index;
                break;
            }
        if (!item)
            throw std::invalid_argument("Model::get_tensor_data: inconsistent m_batch_cache state.");

        Tensor* tensor_ptr =
            item->m_vec->size() == 1 ? &(*item->m_vec)[0] :
            item->m_vec->size() == m_batch_size - 1 ? &(*item->m_vec)[m_batch_index - 1] :
            nullptr;
        if (!tensor_ptr)
            throw std::invalid_argument("Model::get_tensor_data: inconsistent m_batch_cache item state.");

        bool last = m_batch_index == m_batch_size - 1;

        if (item->m_unique && (last || item->m_is_batch))
        {
            t = std::move(*tensor_ptr);
        }
        else
        {
            t = *tensor_ptr;
            if (make_copy)
                t.make_copy_of_data();
        }

        if (!item->m_is_batch)
            return t;
    }
    else if (t.m_type != TensorDataType::none)
    {
        std::string& fn = t.m_name;

        auto lpos = fn.find("_nchw.bin");
        if (lpos == std::string::npos)
        {
            if (required_layout == TensorDataLayout::nhwc)
                throw std::invalid_argument("Model::get_tensor_data: unable to determine tensor data file compatible with required_layout.");
        }
        else
        {
            if (required_layout == TensorDataLayout::nhwc)
            {
                if (t.m_layout != TensorDataLayout::unspecified)
                    throw std::invalid_argument("Model::get_tensor_data: tensor data layout already set.");
                else
                    t.m_layout = TensorDataLayout::nhwc;

                if (t.m_shape.size() != 4)
                    throw std::invalid_argument("Model::get_tensor_data: layout is nhwc but invalid shape.");
                else
                    t.m_shape = { t.m_shape[0], t.m_shape[2], t.m_shape[3], t.m_shape[1] };
            }
            else
            {
                throw std::invalid_argument("Model::get_tensor_data: nchw layout not supported. (not implemented)");
            }

            fn = fn.substr(0, lpos) + "_nhwc.bin";
        }

        load = !m_weights_exclusion_set.contains(fn);

        if (load)
        {
            TensorDataType new_type = get_wp()->get_type_of_next();
            if (new_type != TensorDataType::none) t.m_type = new_type;
        }

        if (!load || !get_wp()->supports_getptr() || make_copy)
        {
            switch (t.m_type)
            {
            case TensorDataType::uint8:
            {
                auto data = load ? get_wp()->get_uint8(fn) : tensor_vector<uint8_t>();
                t.set_vector(std::move(data));
                break;
            }
            case TensorDataType::float16:
            {
                auto data = load ? get_wp()->get_float16(fn) : tensor_vector<uint16_t>();
                t.set_vector(std::move(data));
                break;
            }
            case TensorDataType::float32:
            {
                auto data = load ? get_wp()->get_float32(fn) : tensor_vector<float>();
                t.set_vector(std::move(data));
                break;
            }
            case TensorDataType::int64:
            {
                auto data = load ? get_wp()->get_int64(fn) : tensor_vector<int64_t>();
                t.set_vector(std::move(data));
                break;
            }
            default:
                throw std::invalid_argument("Model::get_tensor_data: unsupported tensor data format.");
            }

            unique = true;
        }
        else
        {
            switch (t.m_type)
            {
            case TensorDataType::uint8:
                t.m_data = get_wp()->getptr_uint8(fn);
                break;
            case TensorDataType::float16:
                t.m_data = get_wp()->getptr_float16(fn);
                break;
            case TensorDataType::float32:
                t.m_data = get_wp()->getptr_float32(fn);
                break;
            case TensorDataType::int64:
                t.m_data = get_wp()->getptr_int64(fn);
                break;
            default:
                throw std::invalid_argument("Model::get_tensor_data: unsupported tensor data format.");
            }

            unique = false;
        }

        t.m_is_static_weights = true;
    }
    else
    {
        Tensor* tensor_ptr = nullptr;
        for (auto& a : m_data)
            if (a.m_name == t.m_name)
            {
                tensor_ptr = &a;
                break;
            }

        if (tensor_ptr == nullptr)
            throw std::invalid_argument("Model::get_tensor_data: input tensor not found: " + t.m_name);

        switch (tensor_ptr->m_type)
        {
        case TensorDataType::float16:
            m_xnnpack->m_cublas_ops.ensure_is_ready(tensor_ptr->get_vector<uint16_t>().data());
            break;
        case TensorDataType::float32:
            m_xnnpack->m_cublas_ops.ensure_is_ready(tensor_ptr->get_vector<float>().data());
            break;
        }

        auto& refs = m_intermediate_refs[t.m_name];

        refs--;

        if (refs < 0)
        {
            throw std::runtime_error("Model::get_tensor_data: inconsistent reference count.");
        }
        else if (refs == 0)
        {
            t = std::move(*tensor_ptr);

            m_data.erase(m_data.begin() + (tensor_ptr - &m_data[0]));

            tensor_ptr = nullptr;

            unique = true;
        }
        else
        {
            if (!make_copy)
            {
                t.m_data = tensor_ptr->m_data;
                t.m_type = tensor_ptr->m_type;
                t.m_layout = tensor_ptr->m_layout;
                t.m_shape = tensor_ptr->m_shape;
                t.m_scale = tensor_ptr->m_scale;
                t.m_zero_point = tensor_ptr->m_zero_point;
                t.m_batch = tensor_ptr->m_batch;
            }
            else
            {
                switch (tensor_ptr->m_type)
                {
                case TensorDataType::uint8:
                    t.set_vector(tensor_vector<uint8_t>(tensor_ptr->get_vector<uint8_t>()));
                    break;
                case TensorDataType::float16:
                    t.set_vector(tensor_vector<uint16_t>(tensor_ptr->get_vector<uint16_t>()));
                    break;
                case TensorDataType::float32:
                    t.set_vector(tensor_vector<float>(tensor_ptr->get_vector<float>()));
                    break;
                case TensorDataType::int64:
                    t.set_vector(tensor_vector<int64_t>(tensor_ptr->get_vector<int64_t>()));
                    break;
                default:
                    throw std::invalid_argument("Model::get_tensor_data: unsupported tensor data format.");
                }

                t.m_layout = tensor_ptr->m_layout;
                t.m_shape = tensor_ptr->m_shape;
                t.m_scale = tensor_ptr->m_scale;
                t.m_zero_point = tensor_ptr->m_zero_point;
                t.m_batch = tensor_ptr->m_batch;
            }

            unique = false;
        }
    }

    if (load)
    {
        if (m_use_fp16_arithmetic && m_requires_upcast && m_ops_queue.size() && m_requires_upcast(m_ops_queue[0].m_type, m_ops_queue[0].m_name))
            requires_float = true;

        size_t size = 0;
        switch (t.m_type)
        {
        case TensorDataType::uint8:
            size = t.get_vector<uint8_t>().size();
            break;
        case TensorDataType::float16:
            size = t.get_vector<uint16_t>().size();
            break;
        case TensorDataType::float32:
            size = t.get_vector<float>().size();
            break;
        case TensorDataType::int64:
            size = t.get_vector<int64_t>().size();
            break;
        }

        int from_shape = 1;
        for (auto& i : t.m_shape)
            from_shape *= i;

        if (from_shape < 0 || from_shape != size)
            throw std::invalid_argument("Model::get_tensor_data: mismatch between tensor shape and data size.");

        bool skip_conversion_fp16 = true;
        if (m_ops_queue.size())
        {
            for (auto& i : m_ops_queue[0].m_input)
                if (i.m_type == TensorDataType::none || i.m_type == TensorDataType::float16)
                {
                    skip_conversion_fp16 = false;
                    break;
                }
        }

        TensorDataType prev_type = t.m_type;

        if (t.m_type == TensorDataType::uint8)
        {
            if (!m_use_uint8_arithmetic)
                dequantize(t, m_use_fp16_arithmetic && !requires_float && !skip_conversion_fp16 ? TensorDataType::float16 : TensorDataType::float32);
        }
        else if (t.m_type == TensorDataType::float16)
        {
            if (!(m_use_fp16_arithmetic && !requires_float))
            {
                auto data = t.get_vector<uint16_t>();
                auto data_fp32 = m_xnnpack->convert<uint16_t, float>(data);
                t.set_vector<float>(std::move(data_fp32));
            }
        }
        else if (t.m_type == TensorDataType::float32)
        {
            if (m_use_fp16_arithmetic && !requires_float && !skip_conversion_fp16)
            {
                auto data = t.get_vector<float>();
                auto data_fp16 = m_xnnpack->convert<float, uint16_t>(data);
                t.set_vector<uint16_t>(std::move(data_fp16));
            }
        }

        if (m_first_run && t.m_is_static_weights && prev_type != t.m_type)
            get_wp()->update(t);

        std::vector<size_t> transpose_perm;

        if (required_layout == TensorDataLayout::nhwc && t.m_layout == TensorDataLayout::unspecified)
        {
            transpose_perm = { 0, 2, 3, 1 };
            if (t.m_shape.size() == 3) // Conv1D
                t.m_shape.push_back(1);
        }
        else if (required_layout == TensorDataLayout::unspecified && t.m_layout == TensorDataLayout::nhwc)
        {
            if (t.m_shape.size() == 3) // Conv1D
                transpose_perm = { 0, 2, 1 };
            else
                transpose_perm = { 0, 3, 1, 2 };
        }

        if (transpose_perm.size())
        {
            if (t.m_shape.size() != transpose_perm.size())
                throw std::invalid_argument("Model::get_tensor_data: transpose required but invalid shape.");

            if (t.m_type == TensorDataType::float32)
            {
                auto result = m_xnnpack->transpose(t.m_shape, t.get_vector<float>(), transpose_perm);
                t.set_vector(std::move(result.second));
                t.m_shape = std::move(result.first);
            }
            else if (t.m_type == TensorDataType::float16)
            {
                auto result = m_xnnpack->transpose(t.m_shape, t.get_vector<uint16_t>(), transpose_perm);
                t.set_vector(std::move(result.second));
                t.m_shape = std::move(result.first);
            }
            else
            {
                auto result = m_xnnpack->transpose(t.m_shape, t.get_vector<uint8_t>(), transpose_perm);
                t.set_vector(std::move(result.second));
                t.m_shape = std::move(result.first);
            }

            t.m_layout = required_layout;
        }
    }

    if (m_batch_size > 1 && m_batch_index == 0)
    {
        if (t.m_batch == nullptr || t.m_batch->size() == 0)
        {
            auto v = std::make_shared<std::vector<Tensor>>();
            v->push_back(t);
            if (make_copy)
            {
                v->back().make_copy_of_data();
                unique = true;
            }
            m_batch_cache.emplace_back(0, unique, false /* is_batch */, std::move(v));
        }
        else
        {
            m_batch_cache.emplace_back(0, unique, true /* is_batch */, t.m_batch);
            t.m_batch.reset();
        }
    }

    return t;
}

void Model::push_tensor(Tensor&& t)
{
    if (m_range_data_calibrate && m_ops_queue.size())
    {
        auto res = get_percentiles(t, 0.001, 0.001);
        if (res)
        {
            auto& name = m_ops_queue[0].m_name;

            if (!m_range_data.count(name))
            {
                m_range_data[name] = *res;
            }
            else
            {
                auto& v = m_range_data[name];

                if (res->first < v.first)
                    v.first = res->first;
                if (res->second > v.second)
                    v.second = res->second;
            }
        }
    }

    bool skip_conversion = false;
    bool force_quantization = false;

    if (!skip_conversion && m_ops_queue.size() >= 2 && m_ops_queue[0].m_output.size() == 1)
    {
        std::string name = m_ops_queue[0].m_output[0].m_name;

        for (auto& input_tensor : m_ops_queue[1].m_input)
            if (input_tensor.m_name == name)
            {
                if (m_intermediate_refs[name] == 1)
                    skip_conversion = true;
                break;
            }
    }

    if (force_quantization || !skip_conversion)
    {
        if (force_quantization || m_use_uint8_qdq || m_use_uint8_arithmetic)
        {
            if (t.m_type != TensorDataType::uint8)
                quantize(t, 0.001, 0.001);
        }
        else if (m_use_fp16_arithmetic && t.m_type == TensorDataType::float32)
        {
            auto data = t.get_vector<float>();
            auto data_fp16 = m_xnnpack->convert<float, uint16_t>(data);
            t.set_vector<uint16_t>(std::move(data_fp16));
        }
    }

    {
        bool pushed = false;

        if ((m_batch_size > 1 && m_batch_index > 0) || !m_ops_queue.size())
            for (auto it = m_data.rbegin(); it != m_data.rend(); ++it)
                if (it->m_name == t.m_name)
                {
                    if (it->m_batch == nullptr)
                        it->m_batch = std::make_shared<std::vector<Tensor>>();
                    it->m_batch->push_back(std::move(t));

                    pushed = true;
                    break;
                }

        if (!pushed)
            m_data.push_back(std::move(t));
    }
}

bool Model::compare_shapes(const std::vector<size_t>& shape_1, const std::vector<size_t>& shape_2, int except /*= -1*/)
{
    if (shape_1.size() != shape_2.size())
        return false;
    else
        for (int i = 0; i < shape_1.size(); i++)
            if (shape_1[i] != shape_2[i])
                if (except == -1 || i != except)
                    return false;

    return true;
}

bool Model::check_output_shape(const std::vector<size_t>& src, std::vector<size_t>& dst)
{
    if (src.size() != dst.size())
    {
        if (m_support_dynamic_shapes && dst.size() == 0)
            dst = src;
        else
            return false;
    }
    else
        for (int i = 0; i < src.size(); i++)
            if (src[i] != dst[i]) {
                if (m_support_dynamic_shapes && dst[i] == 0)
                    dst[i] = src[i];
                else
                    return false;
            }

    return true;
}

bool Model::get_start_and_end(size_t& start, size_t& end, const size_t i, const size_t size, const size_t threads_count)
{
    size_t n = size / threads_count;
    if (!n) n = 1;

    start = i * n;
    end = i >= threads_count - 1 ? size : (i + 1) * n;
    if (start >= end || start >= size || end > size)
        return false;

    return true;
}

std::optional<std::pair<float, float>> Model::get_percentiles(Tensor& input, float from_left, float from_right)
{
    void* data = nullptr;
    size_t size = 0;
    size_t sizeof_element = 0;

    if (input.m_type == TensorDataType::float32)
    {
        if (input.m_type != TensorDataType::float32) throw std::invalid_argument("get_percentiles: wrong data type of input.");

        auto& input_data = input.get_vector<float>();
        size = input_data.size();
        data = input_data.data();

        sizeof_element = sizeof(float);
    }
    else
    {
        if (input.m_type != TensorDataType::float16) throw std::invalid_argument("get_percentiles: wrong data type of input.");

        auto& input_data = input.get_vector<uint16_t>();
        size = input_data.size();
        data = input_data.data();

        sizeof_element = sizeof(uint16_t);
    }

    struct context
    {
        void* data;
        size_t size, threads_count;
        float from_left, from_right, first, second;
        bool set;
        std::mutex mutex;
    };

    context prl_ctx;

    prl_ctx.data = data;
    prl_ctx.size = size;
    prl_ctx.threads_count = m_xnnpack->parallelize_threads_count();
    prl_ctx.from_left = from_left;
    prl_ctx.from_right = from_right;
    prl_ctx.first = +std::numeric_limits<float>::infinity();
    prl_ctx.second = -std::numeric_limits<float>::infinity();
    prl_ctx.set = false;

    void (*pfn)(context*, size_t) = nullptr;

    if (sizeof_element == sizeof(float))
    {
        pfn = [](context* _, size_t i) {

            uint32_t buffer[m_float32_buffer_size];

            size_t start, end;
            if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                return;

            uint32_t* data = (uint32_t*)_->data;

            for (size_t j = start; j < end; j += m_float32_buffer_size)
            {
                size_t n = std::min(end - j, m_float32_buffer_size);
                memcpy(buffer, &data[j], n * sizeof(uint32_t));

                if (auto res = FloatAsUInt::get_percentiles(buffer, buffer + n, n * _->from_left, n * _->from_right))
                {
                    float first = FloatAsUInt::uint32_to_f32(res->first);
                    float second = FloatAsUInt::uint32_to_f32(res->second);

                    {
                        std::lock_guard<std::mutex> guard(_->mutex);

                        if (first < _->first)
                            _->first = first;
                        if (second > _->second)
                            _->second = second;

                        _->set = true;
                    }
                }
            }
        };
    }
    else
    {
        pfn = [](context* _, size_t i) {

            uint16_t buffer[m_float16_buffer_size];

            size_t start, end;
            if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                return;

            uint16_t* data = (uint16_t*)_->data;

            for (size_t j = start; j < end; j += m_float16_buffer_size)
            {
                size_t n = std::min(end - j, m_float16_buffer_size);
                memcpy(buffer, &data[j], n * sizeof(uint16_t));

                if (auto res = FloatAsUInt::get_percentiles(buffer, buffer + n, n * _->from_left, n * _->from_right))
                {
                    float first = FloatAsUInt::f16_to_f32(res->first);
                    float second = FloatAsUInt::f16_to_f32(res->second);

                    {
                        std::lock_guard<std::mutex> guard(_->mutex);

                        if (first < _->first)
                            _->first = first;
                        if (second > _->second)
                            _->second = second;

                        _->set = true;
                    }
                }
            }
        };
    }

    m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.threads_count);

    if (!prl_ctx.set || !std::isfinite(prl_ctx.first) || !std::isfinite(prl_ctx.second) || prl_ctx.first >= prl_ctx.second)
        return std::nullopt;
    else
        return std::make_pair(prl_ctx.first, prl_ctx.second);
}

std::pair<float, uint8_t> Model::range_to_scale(std::pair<float, float> range)
{
    if (range.first > 0 && range.second > 0)
        range.first = 0;
    else if (range.first < 0 && range.second < 0)
        range.second = 0;

    float scale = (range.second - range.first) / 255.0;
    uint8_t zero_point = std::abs(range.first) / scale;

    return std::make_pair(scale, zero_point);
}

bool Model::quantize(Tensor& input, float from_left, float from_right)
{
    auto res = get_percentiles(input, from_left, from_right);
    if (!res)
        return false;

    const auto& [scale, zero_point] = range_to_scale(*res);

    void* data = nullptr;
    size_t size = 0;
    size_t sizeof_element = 0;

    if (input.m_type == TensorDataType::float32)
    {
        if (input.m_type != TensorDataType::float32) throw std::invalid_argument("quantize: wrong data type of input.");

        auto& input_data = input.get_vector<float>();
        size = input_data.size();
        data = input_data.data();

        sizeof_element = sizeof(float);
    }
    else
    {
        if (input.m_type != TensorDataType::float16) throw std::invalid_argument("quantize: wrong data type of input.");

        auto& input_data = input.get_vector<uint16_t>();
        size = input_data.size();
        data = input_data.data();

        sizeof_element = sizeof(uint16_t);
    }

    tensor_vector<uint8_t> output = create_tensor_vector<uint8_t>(size);

    if (sizeof_element == sizeof(float))
    {
        if (!m_xnnpack->convert_qu8<float, uint8_t>((float*)data, output.data(), size, scale, zero_point, false /* single_threaded */))
            throw std::invalid_argument("quantize: convert_qu8 failed.");
    }
    else
    {
        struct context
        {
            void* data;
            uint8_t* output;
            size_t size, threads_count;
            float scale;
            uint8_t zero_point;

            XnnPack* xnnpack;
            bool error;
        };

        context prl_ctx;

        prl_ctx.data = data;
        prl_ctx.output = output.data();
        prl_ctx.size = size;
        prl_ctx.threads_count = m_xnnpack->parallelize_threads_count();
        prl_ctx.scale = scale;
        prl_ctx.zero_point = zero_point;
        prl_ctx.xnnpack = m_xnnpack;
        prl_ctx.error = false;

        void (*pfn)(context*, size_t) = nullptr;

        pfn = [](context* _, size_t i) {

            float buffer[m_float32_buffer_size_w_extra_bytes];

            size_t start, end;
            if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                return;

            uint16_t* data = (uint16_t*)_->data;

            for (size_t j = start; j < end; j += m_float32_buffer_size)
            {
                size_t n = std::min(end - j, m_float32_buffer_size);
                if (!_->xnnpack->convert<uint16_t, float>(&data[j], buffer, n))
                {
                    _->error = true;
                    return;
                }
                if (!_->xnnpack->convert_qu8<float, uint8_t>(buffer, &_->output[j], n, _->scale, _->zero_point))
                {
                    _->error = true;
                    return;
                }
            }
        };

        m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.threads_count);

        if (prl_ctx.error)
            throw std::invalid_argument("quantize: conversion error.");
    }

    input.set_vector<uint8_t>(std::move(output));
    input.m_scale = scale;
    input.m_zero_point = zero_point;

    return true;
}

void Model::dequantize(Tensor& input, TensorDataType dest_type)
{
    if (input.m_type != TensorDataType::uint8)
        throw std::invalid_argument("dequantize: invalid type of input.");

    auto& input_data = input.get_vector<uint8_t>();

    if (dest_type == TensorDataType::float32)
    {
        tensor_vector<float> output_data = create_tensor_vector<float>(input_data.size());

        if (!m_xnnpack->convert_qu8<uint8_t, float>(input_data.data(), output_data.data(), output_data.size(), input.m_scale, input.m_zero_point, false /* single_threaded */))
            throw std::invalid_argument("dequantize: convert_qu8 failed.");

        input.set_vector<float>(std::move(output_data));
    }
    else if (dest_type == TensorDataType::float16)
    {
        tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(input_data.size());

        struct context
        {
            uint8_t* data;
            uint16_t* output;
            size_t size, threads_count;
            float scale;
            uint8_t zero_point;

            XnnPack* xnnpack;
            bool error;
        };

        context prl_ctx;

        prl_ctx.data = input_data.data();
        prl_ctx.output = output_data.data();
        prl_ctx.size = output_data.size();
        prl_ctx.threads_count = m_xnnpack->parallelize_threads_count();
        prl_ctx.scale = input.m_scale;
        prl_ctx.zero_point = input.m_zero_point;
        prl_ctx.xnnpack = m_xnnpack;
        prl_ctx.error = false;

        void (*pfn)(context*, size_t) = nullptr;

        pfn = [](context* _, size_t i) {

            float buffer[m_float32_buffer_size_w_extra_bytes];

            size_t start, end;
            if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                return;

            for (size_t j = start; j < end; j += m_float32_buffer_size)
            {
                size_t n = std::min(end - j, m_float32_buffer_size);
                if (!_->xnnpack->convert_qu8<uint8_t, float>(&_->data[j], buffer, n, _->scale, _->zero_point))
                {
                    _->error = true;
                    return;
                }
                if (!_->xnnpack->convert<float, uint16_t>(buffer, &_->output[j], n))
                {
                    _->error = true;
                    return;
                }
            }
        };

        m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.threads_count);

        if (prl_ctx.error)
            throw std::invalid_argument("dequantize: conversion error.");

        input.set_vector<uint16_t>(std::move(output_data));
    }
    else
        throw std::invalid_argument("dequantize: invalid dest_type.");

    input.m_scale = 0;
    input.m_zero_point = 0;
}

void Model::read_range_data(const char* filename)
{
    auto file = onnxstream::read_file<std::vector<char>>(filename);
    size_t pos = 0;

    while (true)
    {
        auto line = next_file_line(file, pos);
        if (!line.size())
            break;

        auto parts = split_string(line, ",");
        if (parts.size() != 3)
            throw std::invalid_argument("read_range_data: file format error.");

        auto& name = parts[0];
        float start = std::stof(parts[1]);
        float end = std::stof(parts[2]);

        m_range_data[name] = std::make_pair(start, end);
    }
}

void Model::write_range_data(const char* filename)
{
    std::vector<char> data;

    for (const auto& e : m_range_data)
    {
        auto& name = e.first;
        auto start = std::to_string(e.second.first);
        auto end = std::to_string(e.second.second);

        data.insert(data.end(), name.begin(), name.end());
        data.push_back(',');
        data.insert(data.end(), start.begin(), start.end());
        data.push_back(',');
        data.insert(data.end(), end.begin(), end.end());
        data.push_back('\r');
        data.push_back('\n');
    }

    onnxstream::write_file(filename, data);
}

tensor_vector<int64_t> Model::float_to_int64(tensor_vector<float>& input)
{
    size_t s = input.size();
    auto output = create_tensor_vector<int64_t>(s);
    for (size_t i = 0; i < s; i++)
        output[i] = (int64_t)input[i];
    return output;
}

tensor_vector<float> Model::int64_to_float(tensor_vector<int64_t>& input)
{
    size_t s = input.size();
    auto output = create_tensor_vector<float>(s);
    for (size_t i = 0; i < s; i++)
        output[i] = (float)input[i];
    return output;
}

void Model::init()
{
    if (m_intermediate_refs_copy.size() == 0)
    {
        m_pos = 0;

        while (auto op = next_op())
        {
            for (auto& input_tensor : op->m_input)
            {
                if (input_tensor.m_name.size() != 0) {
                    if (input_tensor.m_type == TensorDataType::none)
                        m_intermediate_refs[input_tensor.m_name]++;
                    else
                    {
                        size_t size = 1;
                        for (auto& s : input_tensor.m_shape)
                            size *= s;

                        switch (input_tensor.m_type)
                        {
                        case TensorDataType::uint8: size *= sizeof(uint8_t); break;
                        case TensorDataType::float16: size *= sizeof(uint16_t); break;
                        case TensorDataType::float32: size *= sizeof(float); break;
                        case TensorDataType::int64: size *= sizeof(int64_t); break;
                        default: throw std::invalid_argument("Model::run: unable to calculate tensor size: invalid type.");
                        }

                        get_wp()->on_init(input_tensor.m_type, input_tensor.m_name, size);
                    }
                }
            }
        }

        for (auto& name : m_extra_outputs)
            m_intermediate_refs[name]++;

        m_intermediate_refs_copy = m_intermediate_refs;
    }
    else
    {
        m_intermediate_refs = m_intermediate_refs_copy;

        m_ops_printf_index = 0;

        m_first_run = false;

        get_wp()->on_restart();
    }
}

void Model::run()
{
    init();

    m_pos = 0;

    while (true)
    {
        if (m_ops_queue.size())
            m_ops_queue.erase(m_ops_queue.begin());

        const size_t ops_to_read = 8;

        if (m_ops_queue.size() < ops_to_read)
        {
            while (auto op_opt = next_op())
            {
                m_ops_queue.push_back(std::move(*op_opt));
                if (m_ops_queue.size() >= ops_to_read)
                    break;
            }
        }

        if (!m_ops_queue.size())
            break;

        if (m_fuse_ops_in_attention && m_ops_queue[0].m_type == "MatMul")
        {
            bool with_scale =
                m_ops_queue.size() >= 4 &&
                m_ops_queue[0].m_type == "MatMul" &&
                m_ops_queue[1].m_type == "Mul" &&
                m_ops_queue[2].m_type == "Softmax" &&
                m_ops_queue[3].m_type == "MatMul";

            bool without_scale =
                m_ops_queue.size() >= 3 &&
                m_ops_queue[0].m_type == "MatMul" &&
                m_ops_queue[1].m_type == "Softmax" &&
                m_ops_queue[2].m_type == "MatMul";

            if (with_scale || without_scale)
            {
                Operation& matmul0 = m_ops_queue[0];
                Operation* mul = with_scale ? &m_ops_queue[1] : nullptr;
                Operation& softmax = m_ops_queue[with_scale ? 2 : 1];
                Operation& matmul1 = m_ops_queue[with_scale ? 3 : 2];

                auto is_output_the_first_input = [&](Operation& op0, Operation& op1) {
                    auto& name = op0.m_output[0].m_name;
                    return name == op1.m_input[0].m_name &&
                        m_intermediate_refs[name] == 1;
                };

                if (matmul0.m_input.size() == 2 && matmul0.m_output.size() == 1 &&
                    (!mul || (mul->m_input.size() == 2 && mul->m_output.size() == 1)) &&
                    softmax.m_input.size() == 1 && softmax.m_output.size() == 1 &&
                    softmax.m_attributes.size() == 1 && softmax.m_attributes[0].first == "axis" && softmax.m_attributes[0].second == "-1" &&
                    matmul1.m_input.size() == 2 && matmul1.m_output.size() == 1 &&
                    is_output_the_first_input(matmul0, mul ? *mul : softmax) &&
                    (!mul || is_output_the_first_input(*mul, softmax)) &&
                    is_output_the_first_input(softmax, matmul1))
                {
                    m_intermediate_refs[matmul0.m_output[0].m_name] = 0;
                    if (mul) m_intermediate_refs[mul->m_output[0].m_name] = 0;
                    m_intermediate_refs[softmax.m_output[0].m_name] = 0;

                    Operation op;

                    op.m_name = matmul0.m_name + "_AttentionFusedOps";
                    op.m_type = "AttentionFusedOps";

                    op.m_input.push_back(std::move(matmul0.m_input[0]));
                    op.m_input.push_back(std::move(matmul0.m_input[1]));
                    op.m_input.push_back(mul ? std::move(mul->m_input[1]) : Tensor());
                    op.m_input.push_back(std::move(matmul1.m_input[1]));

                    op.m_output.push_back(std::move(matmul1.m_output[0]));

                    m_ops_queue.erase(m_ops_queue.begin(), m_ops_queue.begin() + (with_scale ? 4 : 3));
                    m_ops_queue.insert(m_ops_queue.begin(), std::move(op));
                }
            }
        }

        if (m_use_scaled_dp_attn_op && m_ops_queue[0].m_type == "Transpose")
        {
            auto is_output_the_input = [&](Operation& op0, Operation& op1, int index = 0) {
                auto& name = op0.m_output[0].m_name;
                return name == op1.m_input[index].m_name &&
                    m_intermediate_refs[name] == 1;
            };

            if (m_ops_queue.size() >= 6 &&
                m_ops_queue[0].m_type == "Transpose" &&
                m_ops_queue[1].m_type == "MatMul" &&
                m_ops_queue[2].m_type == "Div" &&
                m_ops_queue[3].m_type == "Add" &&
                m_ops_queue[4].m_type == "Softmax" &&
                m_ops_queue[5].m_type == "MatMul")
            {
                Operation& transpose = m_ops_queue[0];
                Operation& matmul0 = m_ops_queue[1];
                Operation& div = m_ops_queue[2];
                Operation& add = m_ops_queue[3];
                Operation& softmax = m_ops_queue[4];
                Operation& matmul1 = m_ops_queue[5];

                if (transpose.m_input.size() == 1 && transpose.m_output.size() == 1 &&
                    matmul0.m_input.size() == 2 && matmul0.m_output.size() == 1 &&
                    div.m_input.size() == 2 && div.m_output.size() == 1 &&
                    add.m_input.size() == 2 && add.m_output.size() == 1 &&
                    softmax.m_input.size() == 1 && softmax.m_output.size() == 1 &&
                    softmax.m_attributes.size() == 1 && softmax.m_attributes[0].first == "axis" && softmax.m_attributes[0].second == "-1" &&
                    matmul1.m_input.size() == 2 && matmul1.m_output.size() == 1 &&
                    is_output_the_input(transpose, matmul0, 1) &&
                    is_output_the_input(matmul0, div) &&
                    is_output_the_input(div, add) &&
                    is_output_the_input(add, softmax) &&
                    is_output_the_input(softmax, matmul1))
                {
                    m_intermediate_refs[transpose.m_output[0].m_name] = 0;
                    m_intermediate_refs[matmul0.m_output[0].m_name] = 0;
                    m_intermediate_refs[div.m_output[0].m_name] = 0;
                    m_intermediate_refs[add.m_output[0].m_name] = 0;
                    m_intermediate_refs[softmax.m_output[0].m_name] = 0;

                    Operation op;

                    op.m_name = transpose.m_name + "_ScaledDotProductAttention";
                    op.m_type = "ScaledDotProductAttention";

                    op.m_input.push_back(std::move(matmul0.m_input[0]));
                    op.m_input.push_back(std::move(transpose.m_input[0]));
                    op.m_input.push_back(std::move(div.m_input[1]));
                    op.m_input.push_back(std::move(add.m_input[1]));
                    op.m_input.push_back(std::move(matmul1.m_input[1]));

                    op.m_output.push_back(std::move(matmul1.m_output[0]));

                    m_ops_queue.erase(m_ops_queue.begin(), m_ops_queue.begin() + 6);
                    m_ops_queue.insert(m_ops_queue.begin(), std::move(op));

                    m_scaled_dp_attn_op_used = true;
                }
            }
            else if (m_ops_queue.size() >= 7 &&
                m_ops_queue[0].m_type == "Transpose" &&
                m_ops_queue[1].m_type == "Mul" &&
                m_ops_queue[2].m_type == "Mul" &&
                m_ops_queue[3].m_type == "MatMul" &&
                m_ops_queue[4].m_type == "Add" &&
                m_ops_queue[5].m_type == "Softmax" &&
                m_ops_queue[6].m_type == "MatMul")
            {
                Operation& transpose = m_ops_queue[0];
                Operation& mul0 = m_ops_queue[1];
                Operation& mul1 = m_ops_queue[2];
                Operation& matmul0 = m_ops_queue[3];
                Operation& add = m_ops_queue[4];
                Operation& softmax = m_ops_queue[5];
                Operation& matmul1 = m_ops_queue[6];

                if (transpose.m_input.size() == 1 && transpose.m_output.size() == 1 &&
                    mul0.m_input.size() == 2 && mul0.m_output.size() == 1 &&
                    mul1.m_input.size() == 2 && mul1.m_output.size() == 1 &&
                    matmul0.m_input.size() == 2 && matmul0.m_output.size() == 1 &&
                    add.m_input.size() == 2 && add.m_output.size() == 1 &&
                    softmax.m_input.size() == 1 && softmax.m_output.size() == 1 &&
                    softmax.m_attributes.size() == 1 && softmax.m_attributes[0].first == "axis" && softmax.m_attributes[0].second == "-1" &&
                    matmul1.m_input.size() == 2 && matmul1.m_output.size() == 1 &&
                    is_output_the_input(transpose, mul1) &&
                    is_output_the_input(mul0, matmul0) &&
                    is_output_the_input(mul1, matmul0, 1) &&
                    is_output_the_input(matmul0, add) &&
                    is_output_the_input(add, softmax) &&
                    is_output_the_input(softmax, matmul1))
                {
                    m_intermediate_refs[transpose.m_output[0].m_name] = 0;
                    m_intermediate_refs[mul0.m_output[0].m_name] = 0;
                    m_intermediate_refs[mul1.m_output[0].m_name] = 0;
                    m_intermediate_refs[matmul0.m_output[0].m_name] = 0;
                    m_intermediate_refs[add.m_output[0].m_name] = 0;
                    m_intermediate_refs[softmax.m_output[0].m_name] = 0;

                    Operation op;

                    op.m_name = transpose.m_name + "_ScaledDotProductAttention";
                    op.m_type = "ScaledDotProductAttention";

                    op.m_input.push_back(std::move(mul0.m_input[0]));
                    op.m_input.push_back(std::move(transpose.m_input[0]));
                    op.m_input.push_back(std::move(mul0.m_input[1]));
                    op.m_input.push_back(std::move(add.m_input[1]));
                    op.m_input.push_back(std::move(matmul1.m_input[1]));
                    op.m_input.push_back(std::move(mul1.m_input[1]));

                    op.m_output.push_back(std::move(matmul1.m_output[0]));

                    m_ops_queue.erase(m_ops_queue.begin(), m_ops_queue.begin() + 7);
                    m_ops_queue.insert(m_ops_queue.begin(), std::move(op));

                    m_scaled_dp_attn_op_used = true;
                }
            }
        }

        Operation& op = m_ops_queue[0];

        if (m_ops_printf)
        {
            printf("#%i) %s (%s)\n", m_ops_printf_index++, op.m_type.c_str(), op.m_name.c_str());
        }

        if (m_force_fp16_storage)
        {
            for (auto& t : m_data)
            {
                if (t.m_type == TensorDataType::float32)
                {
                    bool skip_conversion = false;

                    for (auto& input_tensor : op.m_input)
                        if (input_tensor.m_name == t.m_name)
                        {
                            if (m_intermediate_refs[t.m_name] == 1)
                                skip_conversion = true;
                            break;
                        }

                    if (!skip_conversion)
                    {
                        void (*pfn)(Model*, Tensor&) = nullptr;

                        if (m_force_uint8_storage_set.contains(t.m_name))
                        {
                            pfn = [](Model* _, Tensor& t)
                                {
                                    _->quantize(t, 0.001, 0.001);
                                };
                        }
                        else
                        {
                            pfn = [](Model* _, Tensor& t)
                                {
                                    auto data = t.get_vector<float>();
                                    auto data_fp16 = _->m_xnnpack->convert<float, uint16_t>(data);
                                    t.set_vector<uint16_t>(std::move(data_fp16));
                                };
                        }

                        pfn(this, t);
                        if (t.m_batch != nullptr)
                            for (auto& u : *t.m_batch)
                                pfn(this, u);
                    }
                }
            }
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> hrc_now;

        if (m_ops_times_printf)
        {
            hrc_now = std::chrono::high_resolution_clock::now();
        }

        m_batch_size = 1;

        for (auto& t : op.m_input)
        {
            if (t.m_type == TensorDataType::none)
            {
                Tensor* tensor_ptr = nullptr;
                for (auto& a : m_data)
                    if (a.m_name == t.m_name)
                    {
                        tensor_ptr = &a;
                        break;
                    }

                if (tensor_ptr)
                {
                    size_t s = tensor_ptr->m_batch == nullptr ? 1 : tensor_ptr->m_batch->size() + 1;
                    if (s > 1)
                    {
                        if (m_batch_size > 1 && m_batch_size != s)
                            throw std::invalid_argument(op.m_type + ": inconsistent m_batch.size() across two or more tensors.");
                        m_batch_size = s;
                    }
                }
            }
        }

        std::vector<Tensor> outputs_backup;
        bool free_ops_cache = false;

        for (m_batch_index = 0; m_batch_index < m_batch_size; m_batch_index++)
        {
            if (m_batch_size > 1)
            {
                if (m_batch_index == 0)
                    outputs_backup = op.m_output;
                else if (m_batch_index != m_batch_size - 1)
                    op.m_output = outputs_backup;
                else
                    op.m_output = std::move(outputs_backup);
            }

            if (op.m_type == "Unsqueeze")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& data = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& axes = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (axes.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of axes.");

                auto& axes_data = axes.get_vector<int64_t>();

                int rank = data.m_shape.size() + axes_data.size();

                for (auto& a : axes_data)
                {
                    if (a < 0)
                        a = rank + a;
                    if (a < 0 || a >= rank)
                        throw std::invalid_argument(op.m_type + ": wrong data in axes.");
                }

                std::sort(axes_data.begin(), axes_data.end());

                int64_t prev = -1;
                for (auto& a : axes_data)
                {
                    if (a == prev)
                        throw std::invalid_argument(op.m_type + ": duplicate value in axes.");
                    else
                        prev = a;

                    if (a > data.m_shape.size())
                        throw std::invalid_argument(op.m_type + ": wrong data in axes.");

                    data.m_shape.insert(data.m_shape.begin() + a, 1);
                }

                if (!check_output_shape(data.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.m_type = data.m_type;
                output.m_data = std::move(data.m_data);

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Mul")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::unspecified /* required_layout */);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (input_0.m_type == TensorDataType::float32)
                {
                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<float>();
                    auto& input_1_data = input_1.get_vector<float>();

                    auto result = m_xnnpack->multiply(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data());

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input_0.m_type == TensorDataType::int64)
                {
                    if (input_0.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<int64_t>();
                    auto& input_1_data = input_1.get_vector<int64_t>();

                    auto input_0_data_fp32 = int64_to_float(input_0_data);
                    auto input_1_data_fp32 = int64_to_float(input_1_data);

                    auto result = m_xnnpack->multiply(input_0.m_shape, input_0_data_fp32.data(), input_1.m_shape, input_1_data_fp32.data());

                    if (input_0.m_shape.size() == 0 && input_1.m_shape.size() == 0)
                        result.first.clear();

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    auto result_i64 = float_to_int64(result.second);
                    output.set_vector(std::move(result_i64));
                }
                else if (input_0.m_type == TensorDataType::float16)
                {
                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    auto& input_1_data = input_1.get_vector<uint16_t>();

                    auto result = m_xnnpack->multiply(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data());

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input_0.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint8_t>();
                    auto& input_1_data = input_1.get_vector<uint8_t>();

                    XnnPack::Qu8MulData qu8_data;

                    if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                    auto s = range_to_scale(m_range_data[op.m_name]);

                    qu8_data.input1_zero_point = input_0.m_zero_point;
                    qu8_data.input1_scale = input_0.m_scale;
                    qu8_data.input2_zero_point = input_1.m_zero_point;
                    qu8_data.input2_scale = input_1.m_scale;
                    qu8_data.output_zero_point = s.second;
                    qu8_data.output_scale = s.first;

                    auto result = m_xnnpack->multiply<uint8_t>(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data(), nullptr, &qu8_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));

                    output.m_scale = s.first;
                    output.m_zero_point = s.second;
                }

                push_tensor(std::move(output));
            }
            else if (
                op.m_type == "Cos" ||
                op.m_type == "Sin" ||
                op.m_type == "Sqrt" ||
                op.m_type == "Erf")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& output = op.m_output[0];

                size_t sizeof_element = 0;
                void* input_data_data = nullptr;
                size_t input_data_size = 0;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();
                    input_data_size = input_data.size();

                    output.set_vector(std::move(input_data));
                    input_data_data = output.get_vector<float>().data();

                    sizeof_element = sizeof(float);
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();
                    input_data_size = input_data.size();

                    output.set_vector(std::move(input_data));
                    input_data_data = output.get_vector<uint16_t>().data();

                    sizeof_element = sizeof(uint16_t);
                }

                float (*pfn_op)(float) = nullptr;

                if (op.m_type == "Cos")
                {
                    pfn_op = std::cos;
                }
                else if (op.m_type == "Sin")
                {
                    pfn_op = std::sin;
                }
                else if (op.m_type == "Sqrt")
                {
                    pfn_op = std::sqrt;
                }
                else if (op.m_type == "Erf")
                {
                    pfn_op = std::erf;
                }
                else
                    throw std::invalid_argument(op.m_type + ": unrecognized operation.");

                struct context
                {
                    void* data;
                    float (*pfn_op)(float);
                    size_t size, threads_count;

                    XnnPack* xnnpack;
                    bool error;
                };

                context prl_ctx;

                prl_ctx.data = input_data_data;
                prl_ctx.pfn_op = pfn_op;
                prl_ctx.size = input_data_size;
                prl_ctx.threads_count = m_xnnpack->parallelize_threads_count();
                prl_ctx.xnnpack = m_xnnpack;
                prl_ctx.error = false;

                void (*pfn)(context*, size_t) = nullptr;

                if (sizeof_element == sizeof(float))
                {
                    pfn = [](context* _, size_t i) {

                        size_t start, end;
                        if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                            return;

                        float* data = (float*)_->data;
                        for (size_t j = start; j < end; j++)
                            data[j] = _->pfn_op(data[j]);
                    };
                }
                else
                {
                    pfn = [](context* _, size_t i) {

                        float buffer[m_float32_buffer_size_w_extra_bytes];

                        size_t start, end;
                        if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                            return;

                        uint16_t* data = (uint16_t*)_->data;

                        for (size_t j = start; j < end; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(end - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert<uint16_t, float>(&data[j], buffer, n))
                            {
                                _->error = true;
                                return;
                            }

                            for (size_t k = 0; k < n; k++)
                                buffer[k] = _->pfn_op(buffer[k]);

                            if (!_->xnnpack->convert<float, uint16_t>(buffer, &data[j], n))
                            {
                                _->error = true;
                                return;
                            }
                        }
                    };
                }

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.threads_count);

                if (prl_ctx.error)
                    throw std::invalid_argument(op.m_type + ": conversion error between f32 and f16.");

                if (!check_output_shape(input.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Concat")
            {
                if (op.m_input.size() == 0) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& output = op.m_output[0];

                int axis = 0;
                bool axis_found = false;

                for (auto& a : op.m_attributes)
                    if (a.first == "axis")
                    {
                        axis = std::stoi(a.second);
                        axis_found = true;
                        break;
                    }

                if (!axis_found)
                    throw std::invalid_argument(op.m_type + ": axis attribute not found.");

                for (auto& t : op.m_input)
                    if (t.m_shape.size() == 0)
                        t.m_shape = { 1 };

                int rank = op.m_input[0].m_shape.size();

                if (axis < 0)
                    axis = rank + axis;
                if (axis < 0 || axis >= rank)
                    throw std::invalid_argument(op.m_type + ": invalid axis attribute.");

                std::vector<std::pair<void*, size_t>> inputs;
                size_t final_dim = 0;
                size_t output_stride = 0;

                size_t sizeof_element = 0;

                for (auto& t : op.m_input)
                {
                    auto& input = get_tensor_data(t);

                    if (!compare_shapes(t.m_shape, op.m_input[0].m_shape, axis))
                        throw std::invalid_argument(op.m_type + ": invalid shape of one or more inputs.");

                    size_t num = 1;
                    for (int i = axis; i < t.m_shape.size(); i++)
                        num *= t.m_shape[i];

                    final_dim += t.m_shape[axis];
                    output_stride += num;

                    if (!sizeof_element)
                    {
                        if (input.m_type == TensorDataType::float16)
                            sizeof_element = sizeof(uint16_t);
                        else if (input.m_type == TensorDataType::float32)
                            sizeof_element = sizeof(float);
                        else if (input.m_type == TensorDataType::int64)
                            sizeof_element = sizeof(int64_t);
                        else
                            throw std::invalid_argument(op.m_type + ": wrong data type of input.");
                    }
                    else
                    {
                        if (
                            (input.m_type != TensorDataType::float16 && input.m_type != TensorDataType::float32 && input.m_type != TensorDataType::int64) ||
                            (input.m_type == TensorDataType::float16 && sizeof_element != sizeof(uint16_t)) ||
                            (input.m_type == TensorDataType::float32 && sizeof_element != sizeof(float)) ||
                            (input.m_type == TensorDataType::int64 && sizeof_element != sizeof(int64_t))
                            )
                        {
                            throw std::invalid_argument(op.m_type + ": wrong data type of input.");
                        }
                    }

                    if (input.m_type == TensorDataType::float32)
                    {
                        auto& input_data = input.get_vector<float>();
                        inputs.emplace_back(input_data.data(), num);
                    }
                    else if (input.m_type == TensorDataType::float16)
                    {
                        auto& input_data = input.get_vector<uint16_t>();
                        inputs.emplace_back(input_data.data(), num);
                    }
                    else
                    {
                        auto& input_data = input.get_vector<int64_t>();
                        inputs.emplace_back(input_data.data(), num);
                    }
                }

                std::vector<size_t> output_shape(op.m_input[0].m_shape);
                output_shape[axis] = final_dim;

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                void* ptr_output = nullptr;

                if (sizeof_element == sizeof(float))
                {
                    tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);
                    output.set_vector(std::move(output_data));
                    ptr_output = output.get_vector<float>().data();
                }
                else if (sizeof_element == sizeof(uint16_t))
                {
                    tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);
                    output.set_vector(std::move(output_data));
                    ptr_output = output.get_vector<uint16_t>().data();
                }
                else
                {
                    tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);
                    output.set_vector(std::move(output_data));
                    ptr_output = output.get_vector<int64_t>().data();
                }

                size_t ops_num = output_num_els / output_stride;

                struct context
                {
                    std::vector<std::pair<void*, size_t>> inputs;
                    size_t output_stride, sizeof_element;
                    void* ptr_output;
                };

                context prl_ctx;

                prl_ctx.inputs = std::move(inputs);

                prl_ctx.output_stride = output_stride;
                prl_ctx.sizeof_element = sizeof_element;
                prl_ctx.ptr_output = ptr_output;

                void (*pfn)(context*, size_t) = [](context* _, size_t i)
                {
                    size_t index_output = _->output_stride * i;

                    for (const auto& p : _->inputs)
                    {
                        memcpy(
                            (uint8_t*)_->ptr_output + index_output * _->sizeof_element,
                            (uint8_t*)p.first + p.second * i * _->sizeof_element,
                            p.second * _->sizeof_element);

                        index_output += p.second;
                    }
                };

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, ops_num);

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Gemm")
            {
                if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs. 2 inputs case not implemented.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                float alpha = 1;
                float beta = 1;
                int transA = 0;
                int transB = 0;

                for (auto& a : op.m_attributes)
                    if (a.first == "alpha")
                        alpha = std::stof(a.second);
                    else if (a.first == "beta")
                        beta = std::stof(a.second);
                    else if (a.first == "transA")
                        transA = std::stoi(a.second);
                    else if (a.first == "transB")
                        transB = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (alpha != 1)
                    throw std::invalid_argument(op.m_type + ": alpha != 1 case not implemented.");
                if (beta != 1)
                    throw std::invalid_argument(op.m_type + ": beta != 1 case not implemented.");
                if (transA != 0)
                    throw std::invalid_argument(op.m_type + ": transA != 0 case not implemented.");
                if (transB != 0)
                    throw std::invalid_argument(op.m_type + ": transB != 0 case not implemented.");

                auto& input_0 = get_tensor_data(op.m_input[0]);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& bias = get_tensor_data(op.m_input[2]);
                auto& output = op.m_output[0];

                if (bias.m_shape.size() == 1)
                    bias.m_shape.insert(bias.m_shape.begin(), 1);

                if (input_0.m_type == TensorDataType::float32)
                {
                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                    if (bias.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of bias.");

                    auto& input_0_data = input_0.get_vector<float>();
                    auto& input_1_data = input_1.get_vector<float>();
                    auto& bias_data = bias.get_vector<float>();

                    auto result = m_xnnpack->matrix_multiply<float, float>(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data(), &bias.m_shape, bias_data.data());

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                    if (bias.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of bias.");

                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    auto& input_1_data = input_1.get_vector<uint16_t>();
                    auto& bias_data = bias.get_vector<uint16_t>();

                    auto result = m_xnnpack->matrix_multiply<uint16_t, uint16_t>(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data(), &bias.m_shape, bias_data.data());

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Sigmoid")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], m_use_uint8_arithmetic /* make_copy */);
                auto& output = op.m_output[0];

                if ((input.m_type == TensorDataType::uint8) != m_use_uint8_arithmetic) throw std::invalid_argument(op.m_type + ": make_copy == true and tensor_type != uint8 or viceversa.");

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    auto result = m_xnnpack->sigmoid(input.m_shape, input_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    auto result = m_xnnpack->sigmoid(input.m_shape, input_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint8_t>();
                    output.set_vector(std::move(input_data));

                    if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                    auto s = range_to_scale(m_range_data[op.m_name]);
                    output.m_zero_point = s.second;
                    output.m_scale = s.first;

                    struct context
                    {
                        uint8_t* data;
                        size_t size, threads_count;

                        XnnPack* xnnpack;
                        bool error;

                        uint8_t input_zero_point;
                        float input_scale;
                        uint8_t output_zero_point;
                        float output_scale;
                    };

                    context prl_ctx;

                    prl_ctx.data = output.get_vector<uint8_t>().data();
                    prl_ctx.size = output.get_vector<uint8_t>().size();
                    prl_ctx.threads_count = m_xnnpack->parallelize_threads_count();

                    prl_ctx.xnnpack = m_xnnpack;
                    prl_ctx.error = false;

                    prl_ctx.input_zero_point = input.m_zero_point;
                    prl_ctx.input_scale = input.m_scale;
                    prl_ctx.output_zero_point = output.m_zero_point;
                    prl_ctx.output_scale = output.m_scale;

                    void (*pfn)(context*, size_t) = nullptr;

                    pfn = [](context* _, size_t i) {

                        float buffer[m_float32_buffer_size_w_extra_bytes];

                        size_t start, end;
                        if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                            return;

                        for (size_t j = start; j < end; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(end - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert_qu8<uint8_t, float>(&_->data[j], buffer, n, _->input_scale, _->input_zero_point))
                            {
                                _->error = true;
                                return;
                            }

                            for (size_t k = 0; k < n; k++)
                                buffer[k] = 1 / (1 + std::exp(-buffer[k]));

                            if (!_->xnnpack->convert_qu8<float, uint8_t>(buffer, &_->data[j], n, _->output_scale, _->output_zero_point))
                            {
                                _->error = true;
                                return;
                            }
                        }
                    };

                    m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.threads_count);

                    if (prl_ctx.error)
                        throw std::invalid_argument(op.m_type + ": quantization error.");

                    if (!check_output_shape(input.m_shape, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Conv")
            {
                if (op.m_input.size() != 3 && op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                std::vector<int> dilations, kernel_shape, pads, strides;
                int group = 1;

                for (auto& a : op.m_attributes)
                    if (a.first == "dilations")
                        dilations = string_to_int_vec<int>(a.second);
                    else if (a.first == "group")
                        group = std::stoi(a.second);
                    else if (a.first == "kernel_shape")
                        kernel_shape = string_to_int_vec<int>(a.second);
                    else if (a.first == "pads")
                        pads = string_to_int_vec<int>(a.second);
                    else if (a.first == "strides")
                        strides = string_to_int_vec<int>(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                auto& x = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, m_use_nchw_convs ? TensorDataLayout::unspecified : TensorDataLayout::nhwc /* required_layout */);
                auto& w = get_tensor_data(op.m_input[1], false /* make_copy */, false /* requires_float */, m_use_nchw_convs ? TensorDataLayout::unspecified : TensorDataLayout::nhwc /* required_layout */);
                auto* b = op.m_input.size() > 2 ? &get_tensor_data(op.m_input[2], true /* make_copy */) : nullptr;
                auto& output = op.m_output[0];

                bool is1D = dilations.size() == 1;

                if (is1D)
                {
                    dilations.push_back(1);
                    kernel_shape.push_back(1);
                    if (pads.size() != 2)
                    {
                        throw std::invalid_argument(op.m_type + ": invalid pads attribute value.");
                    }
                    else
                    {
                        pads.insert(pads.begin() + 1, 0);
                        pads.push_back(0);
                    }
                    if (strides.size() != 1)
                    {
                        throw std::invalid_argument(op.m_type + ": invalid strides attribute value.");
                    }
                    else
                    {
                        strides.push_back(strides[0]);
                    }
                }

                if (!are_all_equal(dilations, { 1, 1 }))
                    throw std::invalid_argument(op.m_type + ": invalid dilations attribute value (not implemented).");
                /*if (!are_all_equal(pads, {1, 1, 1, 1}))
                    throw std::invalid_argument(op.m_type + ": invalid pads attribute value (not implemented).");*/
                /*if (!are_all_equal(strides, {1, 1}))
                    throw std::invalid_argument(op.m_type + ": invalid strides attribute value (not implemented).");*/
                if (group != 1)
                    throw std::invalid_argument(op.m_type + ": invalid group attribute value (not implemented).");

                std::string cache_key;
                if (m_use_ops_cache && w.m_is_static_weights && (!b || b->m_is_static_weights))
                {
                    cache_key = op.m_name;
                    if (m_first_run && m_batch_index == 0)
                    {
                        m_weights_exclusion_set.insert(w.m_name);
                        get_wp()->remove(w.m_name);

                        if (b)
                        {
                            m_weights_exclusion_set.insert(b->m_name);
                            get_wp()->remove(b->m_name);
                        }
                    }
                }
                else if (m_batch_size > 1 && w.m_is_static_weights && (!b || b->m_is_static_weights))
                {
                    cache_key = op.m_name;
                    free_ops_cache = true;
                }

                if (x.m_layout != (m_use_nchw_convs ? TensorDataLayout::unspecified : TensorDataLayout::nhwc)) throw std::invalid_argument(op.m_type + ": wrong layout of X.");
                if (w.m_layout != (m_use_nchw_convs ? TensorDataLayout::unspecified : TensorDataLayout::nhwc)) throw std::invalid_argument(op.m_type + ": wrong layout of W.");

                if (w.m_shape.size() != 4 || !are_all_equal(kernel_shape, { w.m_shape[1], w.m_shape[2] }))
                    throw std::invalid_argument(op.m_type + ": invalid shape of W or invalid kernel_shape (not implemented?).");

                std::vector<size_t> result_first;

                if (x.m_type == TensorDataType::float32)
                {
                    if (x.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of X.");
                    if (w.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of W.");
                    if (b && b->m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of B.");

                    auto& x_data = x.get_vector<float>();
                    auto& w_data = w.get_vector<float>();
                    auto* b_data = b ? &b->get_vector<float>() : nullptr;

                    auto result = m_xnnpack->convolution<float, float>(
                        m_use_nchw_convs /* nchw */,
                        x.m_shape, x_data,
                        w.m_shape, w_data,
                        b_data ? b_data->data() : nullptr, b_data ? b_data->size() : 0,
                        dilations, kernel_shape, pads, strides, group,
                        nullptr /* qu8_data */, cache_key);

                    result_first = std::move(result.first);

                    output.set_vector(std::move(result.second));
                }
                else if (x.m_type == TensorDataType::float16)
                {
                    if (x.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of X.");
                    if (w.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of W.");
                    if (b && b->m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of B.");

                    auto& x_data = x.get_vector<uint16_t>();
                    auto& w_data = w.get_vector<uint16_t>();
                    auto* b_data = b ? &b->get_vector<uint16_t>() : nullptr;

                    auto result = m_xnnpack->convolution<uint16_t, uint16_t>(
                        m_use_nchw_convs /* nchw */,
                        x.m_shape, x_data,
                        w.m_shape, w_data,
                        b_data ? b_data->data() : nullptr, b_data ? b_data->size() : 0,
                        dilations, kernel_shape, pads, strides, group,
                        nullptr /* qu8_data */, cache_key);

                    result_first = std::move(result.first);

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (x.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of X.");
                    if (w.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of W.");
                    if (b && b->m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of B.");

                    auto& x_data = x.get_vector<uint8_t>();
                    auto& w_data = w.get_vector<uint8_t>();
                    auto* b_data = b ? &b->get_vector<float>() : nullptr;

                    if (b_data)
                    {
                        struct context
                        {
                            float* ptr;
                            float scale;
                        };

                        context prl_ctx;

                        prl_ctx.ptr = b_data->data();
                        prl_ctx.scale = x.m_scale * w.m_scale;

                        void (*pfn)(context*, size_t) = nullptr;

                        pfn = [](context* _, size_t i)
                            {
                                *(int32_t*)(_->ptr + i) = (int32_t)(_->ptr[i] / _->scale);
                            };

                        m_xnnpack->parallelize((void*)pfn, &prl_ctx, b_data->size());
                    }

                    XnnPack::Qu8ConvData qu8_data;

                    if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                    auto s = range_to_scale(m_range_data[op.m_name]);

                    qu8_data.input_zero_point = x.m_zero_point;
                    qu8_data.input_scale = x.m_scale;
                    qu8_data.kernel_zero_point = w.m_zero_point;
                    qu8_data.kernel_scale = w.m_scale;
                    qu8_data.output_zero_point = s.second;
                    qu8_data.output_scale = s.first;

                    auto result = m_xnnpack->convolution<uint8_t, int32_t>(
                        m_use_nchw_convs /* nchw */,
                        x.m_shape, x_data,
                        w.m_shape, w_data,
                        b_data ? (int32_t*)b_data->data() : nullptr, b_data ? b_data->size() : 0,
                        dilations, kernel_shape, pads, strides, group,
                        &qu8_data, cache_key);

                    result_first = std::move(result.first);

                    output.set_vector(std::move(result.second));

                    output.m_scale = s.first;
                    output.m_zero_point = s.second;
                }

                if (is1D)
                    output.m_shape.push_back(1);

                if (result_first.size() != 4 ||
                    !check_output_shape({ result_first[0], result_first[3], result_first[1], result_first[2] }, output.m_shape))
                {
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
                }

                if (is1D)
                    result_first.erase(result_first.begin() + (m_use_nchw_convs ? 3 : 2));

                output.m_layout = m_use_nchw_convs ? TensorDataLayout::unspecified : TensorDataLayout::nhwc;
                output.m_shape = std::move(result_first);

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Reshape")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                int allowzero = 0;

                for (auto& a : op.m_attributes)
                    if (a.first == "allowzero")
                        allowzero = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (allowzero)
                    throw std::invalid_argument(op.m_type + ": allowzero must be 0 (not implemented).");

                auto& data = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& shape = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (shape.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of shape.");

                auto& shape_data = shape.get_vector<int64_t>();

                if (shape_data.size() == 0)
                    throw std::invalid_argument(op.m_type + ": size of shape must be non-0 (not implemented).");

                std::vector<size_t> output_shape;
                for (int i = 0; i < shape_data.size(); i++)
                {
                    auto& d = shape_data[i];

                    if (d == 0)
                    {
                        if (i >= data.m_shape.size())
                            throw std::invalid_argument(op.m_type + ": insufficient number of dimensions in shape of data.");

                        d = data.m_shape[i];
                    }

                    output_shape.push_back(d);
                }

                size_t data_elements = 1;
                for (auto& d : data.m_shape)
                    data_elements *= d;

                bool minus_1_found = false;
                for (auto& d : output_shape)
                {
                    if (d == -1)
                    {
                        if (minus_1_found)
                            throw std::invalid_argument(op.m_type + ": more than one -1 in shape.");
                        else
                            minus_1_found = true;

                        size_t others = 1;
                        for (auto& d2 : output_shape)
                            if (d2 != -1)
                                others *= d2;

                        if (data_elements < others || data_elements % others)
                            throw std::invalid_argument(op.m_type + ": unable to infer dimension of output shape.");

                        d = data_elements / others;
                    }
                }

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.m_type = data.m_type;
                output.m_data = std::move(data.m_data);

                output.m_scale = data.m_scale;
                output.m_zero_point = data.m_zero_point;

                push_tensor(std::move(output));
            }
            else if (op.m_type == "InstanceNormalization")
            {
                if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                float epsilon = 1e-05;

                for (auto& a : op.m_attributes)
                    if (a.first == "epsilon")
                        epsilon = std::stof(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                auto& input = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& scale = get_tensor_data(op.m_input[1], false /* make_copy */, true /* requires_float */);
                auto& b = get_tensor_data(op.m_input[2], false /* make_copy */, true /* requires_float */);
                auto& output = op.m_output[0];

                if (scale.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of scale.");
                if (b.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of b.");

                if (input.m_shape.size() != 3 || input.m_shape[0] != 1)
                    throw std::invalid_argument(op.m_type + ": shape of input must have 3 dimensions and first dimension must be 1 (n-dimensions case not implemented).");

                size_t channels = input.m_shape[1];
                size_t elements_per_c = input.m_shape[2];

                if (scale.m_shape.size() != 1 || scale.m_shape[0] != channels)
                    throw std::invalid_argument(op.m_type + ": shape of scale must be (C).");
                if (b.m_shape.size() != 1 || b.m_shape[0] != channels)
                    throw std::invalid_argument(op.m_type + ": shape of b must be (C).");

                auto& scale_data = scale.get_vector<float>();
                auto& b_data = b.get_vector<float>();

                void* input_data_data = nullptr;
                size_t sizeof_element = 0;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();
                    output.set_vector(std::move(input_data));

                    input_data_data = output.get_vector<float>().data();
                    sizeof_element = sizeof(float);
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();
                    output.set_vector(std::move(input_data));

                    input_data_data = output.get_vector<uint16_t>().data();
                    sizeof_element = sizeof(uint16_t);
                }
                else
                {
                    if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint8_t>();
                    output.set_vector(std::move(input_data));

                    input_data_data = output.get_vector<uint8_t>().data();
                    sizeof_element = sizeof(uint8_t);

                    if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                    auto s = range_to_scale(m_range_data[op.m_name]);
                    output.m_zero_point = s.second;
                    output.m_scale = s.first;
                }

                struct context
                {
                    void* input_data_ptr;
                    float* scale_data_ptr, * b_data_ptr;
                    size_t elements_per_c;
                    float epsilon;

                    XnnPack* xnnpack;
                    bool error;

                    uint8_t input_zero_point;
                    float input_scale;
                    uint8_t output_zero_point;
                    float output_scale;
                };

                context prl_ctx;

                prl_ctx.input_data_ptr = input_data_data;
                prl_ctx.scale_data_ptr = scale_data.data();
                prl_ctx.b_data_ptr = b_data.data();
                prl_ctx.elements_per_c = elements_per_c;
                prl_ctx.epsilon = epsilon;
                prl_ctx.xnnpack = m_xnnpack;
                prl_ctx.error = false;

                if (sizeof_element == sizeof(uint8_t))
                {
                    prl_ctx.input_zero_point = input.m_zero_point;
                    prl_ctx.input_scale = input.m_scale;
                    prl_ctx.output_zero_point = output.m_zero_point;
                    prl_ctx.output_scale = output.m_scale;
                }

                void (*pfn)(context*, size_t) = nullptr;

                if (sizeof_element == sizeof(float))
                {
                    pfn = [](context* _, size_t i)
                    {
                        float* data = (float*)_->input_data_ptr + _->elements_per_c * i;
                        float scale = _->scale_data_ptr[i];
                        float b = _->b_data_ptr[i];

                        double mean = 0;
                        for (int j = 0; j < _->elements_per_c; j++)
                            mean += data[j];
                        mean /= _->elements_per_c;

                        double variance = 0;
                        for (int j = 0; j < _->elements_per_c; j++)
                        {
                            float dev = data[j] - mean;
                            variance += dev * dev;
                        }
                        variance /= _->elements_per_c;

                        double sr = std::sqrt(variance + _->epsilon);
                        for (int j = 0; j < _->elements_per_c; j++)
                            data[j] = scale * (data[j] - mean) / sr + b;
                    };
                }
                else if (sizeof_element == sizeof(uint16_t))
                {
                    pfn = [](context* _, size_t i)
                    {
                        float buffer[m_float32_buffer_size_w_extra_bytes];

                        uint16_t* data = (uint16_t*)_->input_data_ptr + _->elements_per_c * i;
                        float scale = _->scale_data_ptr[i];
                        float b = _->b_data_ptr[i];

                        double mean = 0;
                        for (int j = 0; j < _->elements_per_c; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->elements_per_c - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert<uint16_t, float>(&data[j], buffer, n))
                            {
                                _->error = true;
                                return;
                            }
                            for (size_t k = 0; k < n; k++)
                                mean += buffer[k];
                        }
                        mean /= _->elements_per_c;

                        double variance = 0;
                        for (int j = 0; j < _->elements_per_c; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->elements_per_c - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert<uint16_t, float>(&data[j], buffer, n))
                            {
                                _->error = true;
                                return;
                            }
                            for (size_t k = 0; k < n; k++)
                            {
                                float dev = buffer[k] - mean;
                                variance += dev * dev;
                            }
                        }
                        variance /= _->elements_per_c;

                        for (int j = 0; j < _->elements_per_c; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->elements_per_c - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert<uint16_t, float>(&data[j], buffer, n))
                            {
                                _->error = true;
                                return;
                            }
                            double sr = std::sqrt(variance + _->epsilon);
                            for (size_t k = 0; k < n; k++)
                                buffer[k] = scale * (buffer[k] - mean) / sr + b;
                            if (!_->xnnpack->convert<float, uint16_t>(buffer, &data[j], n))
                            {
                                _->error = true;
                                return;
                            }
                        }
                    };
                }
                else
                {
                    pfn = [](context* _, size_t i)
                    {
                        float buffer[m_float32_buffer_size_w_extra_bytes];

                        uint8_t* data = (uint8_t*)_->input_data_ptr + _->elements_per_c * i;
                        float scale = _->scale_data_ptr[i];
                        float b = _->b_data_ptr[i];

                        double mean = 0;
                        for (int j = 0; j < _->elements_per_c; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->elements_per_c - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert_qu8<uint8_t, float>(&data[j], buffer, n, _->input_scale, _->input_zero_point))
                            {
                                _->error = true;
                                return;
                            }
                            for (size_t k = 0; k < n; k++)
                                mean += buffer[k];
                        }
                        mean /= _->elements_per_c;

                        double variance = 0;
                        for (int j = 0; j < _->elements_per_c; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->elements_per_c - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert_qu8<uint8_t, float>(&data[j], buffer, n, _->input_scale, _->input_zero_point))
                            {
                                _->error = true;
                                return;
                            }
                            for (size_t k = 0; k < n; k++)
                            {
                                float dev = buffer[k] - mean;
                                variance += dev * dev;
                            }
                        }
                        variance /= _->elements_per_c;

                        for (int j = 0; j < _->elements_per_c; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->elements_per_c - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert_qu8<uint8_t, float>(&data[j], buffer, n, _->input_scale, _->input_zero_point))
                            {
                                _->error = true;
                                return;
                            }
                            double sr = std::sqrt(variance + _->epsilon);
                            for (size_t k = 0; k < n; k++)
                                buffer[k] = scale * (buffer[k] - mean) / sr + b;
                            if (!_->xnnpack->convert_qu8<float, uint8_t>(buffer, &data[j], n, _->output_scale, _->output_zero_point))
                            {
                                _->error = true;
                                return;
                            }
                        }
                    };
                }

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, channels);

                if (prl_ctx.error)
                    throw std::invalid_argument(op.m_type + ": conversion error between f32 and f16.");

                if (!check_output_shape(input.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Add")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0]);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (input_0.m_type == TensorDataType::float32)
                {
                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<float>();
                    auto& input_1_data = input_1.get_vector<float>();

                    auto result = m_xnnpack->add(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input_0.m_type == TensorDataType::float16)
                {
                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    auto& input_1_data = input_1.get_vector<uint16_t>();

                    auto result = m_xnnpack->add(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input_0.m_type == TensorDataType::uint8)
                {
                    if (input_0.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint8_t>();
                    auto& input_1_data = input_1.get_vector<uint8_t>();

                    XnnPack::Qu8AddData qu8_data;

                    if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                    auto s = range_to_scale(m_range_data[op.m_name]);

                    qu8_data.input1_zero_point = input_0.m_zero_point;
                    qu8_data.input1_scale = input_0.m_scale;
                    qu8_data.input2_zero_point = input_1.m_zero_point;
                    qu8_data.input2_scale = input_1.m_scale;
                    qu8_data.output_zero_point = s.second;
                    qu8_data.output_scale = s.first;

                    auto result = m_xnnpack->add(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data, &qu8_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));

                    output.m_scale = s.first;
                    output.m_zero_point = s.second;
                }
                else
                {
                    if (input_0.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<int64_t>();
                    auto& input_1_data = input_1.get_vector<int64_t>();

                    std::vector<size_t> output_shape;
                    tensor_vector<int64_t> output_data;

                    bool input_1_scalar = input_1.m_shape.size() == 0;

                    if (input_1_scalar && input_0.m_shape.size() == 0)
                    {
                        size_t output_num_els = 1;

                        output_data = create_tensor_vector<int64_t>(output_num_els);

                        output_data[0] = input_0_data[0] + input_1_data[0];
                    }
                    else if (input_1_scalar && input_0.m_shape.size() == 1)
                    {
                        size_t output_num_els = input_0.m_shape[0];
                        output_shape = { output_num_els };

                        output_data = create_tensor_vector<int64_t>(output_num_els);

                        for (size_t i = 0; i < output_num_els; i++)
                            output_data[i] = input_0_data[i] + input_1_data[0];
                    }
                    else
                    {
                        auto input_0_data_fp32 = int64_to_float(input_0_data);
                        auto input_1_data_fp32 = int64_to_float(input_1_data);

                        auto result = m_xnnpack->add(input_0.m_shape, input_0_data_fp32, input_1.m_shape, input_1_data_fp32);

                        output_shape = std::move(result.first);
                        output_data = float_to_int64(result.second);
                    }

                    if (!check_output_shape(output_shape, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(output_data));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Transpose")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                std::vector<size_t> perm;

                for (auto& a : op.m_attributes)
                    if (a.first == "perm")
                        perm = string_to_int_vec<size_t>(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& output = op.m_output[0];

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    auto result = m_xnnpack->transpose(input.m_shape, input_data, perm);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    auto result = m_xnnpack->transpose(input.m_shape, input_data, perm);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint8_t>();

                    auto result = m_xnnpack->transpose(input.m_shape, input_data, perm);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));

                    output.m_scale = input.m_scale;
                    output.m_zero_point = input.m_zero_point;
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "ReduceMean")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], false /* make_copy */);
                auto& output = op.m_output[0];

                if (input.m_shape.size() == 0)
                    throw std::invalid_argument(op.m_type + ": invalid shape of input.");

                std::vector<int> axes;
                int keepdims = 1;

                for (auto& a : op.m_attributes)
                    if (a.first == "axes")
                        axes = string_to_int_vec<int>(a.second);
                    else if (a.first == "keepdims")
                        keepdims = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (axes.size() != 1)
                    throw std::invalid_argument(op.m_type + ": reduce supported on 1 axis only (not implemented).");
                if (keepdims != 1)
                    throw std::invalid_argument(op.m_type + ": 'keepdims' must be 1 (not implemented).");

                int axis = axes[0];

                if (axis < 0)
                    axis = (int)input.m_shape.size() + axis;
                if (axis < 0 || axis >= input.m_shape.size())
                    throw std::invalid_argument(op.m_type + ": wrong data in axes.");

                if (axis != input.m_shape.size() - 1)
                    throw std::invalid_argument(op.m_type + ": reduce supported on last axis only (not implemented).");

                std::vector<size_t> output_shape(input.m_shape);

                size_t len = output_shape.back();
                output_shape.back() = 1;

                size_t output_size = 1;
                for (auto& s : output_shape)
                    output_size *= s;

                void* input_data_data = nullptr;
                void* output_data_data = nullptr;
                size_t sizeof_element = 0;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();
                    input_data_data = input_data.data();

                    tensor_vector<float> output_data = create_tensor_vector<float>(output_size);
                    output.set_vector(std::move(output_data));
                    output_data_data = output.get_vector<float>().data();

                    sizeof_element = sizeof(float);
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();
                    input_data_data = input_data.data();

                    tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_size);
                    output.set_vector(std::move(output_data));
                    output_data_data = output.get_vector<uint16_t>().data();

                    sizeof_element = sizeof(uint16_t);
                }

                struct context
                {
                    void* input_data_ptr, * output_data_ptr;
                    size_t len;

                    XnnPack* xnnpack;
                    bool error;
                };

                context prl_ctx;

                prl_ctx.input_data_ptr = input_data_data;
                prl_ctx.output_data_ptr = output_data_data;
                prl_ctx.len = len;
                prl_ctx.xnnpack = m_xnnpack;
                prl_ctx.error = false;

                void (*pfn)(context*, size_t) = nullptr;

                if (sizeof_element == sizeof(float))
                {
                    pfn = [](context* _, size_t i)
                    {
                        float* ptr_in = (float*)_->input_data_ptr + i * _->len;
                        float* ptr_out = (float*)_->output_data_ptr + i;

                        double mean = 0;
                        for (size_t j = 0; j < _->len; j++)
                            mean += *ptr_in++;
                        mean /= _->len;

                        *ptr_out = mean;
                    };
                }
                else
                {
                    pfn = [](context* _, size_t i)
                    {
                        float buffer[m_float32_buffer_size_w_extra_bytes];

                        uint16_t* ptr_in = (uint16_t*)_->input_data_ptr + i * _->len;
                        uint16_t* ptr_out = (uint16_t*)_->output_data_ptr + i;

                        double mean = 0;
                        for (size_t j = 0; j < _->len; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(_->len - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert<uint16_t, float>(&ptr_in[j], buffer, n))
                            {
                                _->error = true;
                                return;
                            }

                            for (size_t k = 0; k < n; k++)
                            {
                                float y = buffer[k];
                                mean += y;
                            }
                        }
                        mean /= _->len;

                        float mean_f = mean;
                        if (!_->xnnpack->convert<float, uint16_t>(&mean_f, ptr_out, 1))
                        {
                            _->error = true;
                            return;
                        }
                    };
                }

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, output_size);

                if (prl_ctx.error)
                    throw std::invalid_argument(op.m_type + ": conversion error between f32 and f16.");

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Sub")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0]);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (input_0.m_type == TensorDataType::float32)
                {
                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<float>();
                    auto& input_1_data = input_1.get_vector<float>();

                    auto result = m_xnnpack->subtract(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input_0.m_type == TensorDataType::float16)
                {
                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    auto& input_1_data = input_1.get_vector<uint16_t>();

                    auto result = m_xnnpack->subtract(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input_0.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    if (input_1.m_shape.size() != 0)
                        throw std::invalid_argument(op.m_type + ": second input must be a scalar (not implemented).");

                    auto& input_0_data = input_0.get_vector<int64_t>();
                    auto& input_1_data = input_1.get_vector<int64_t>();

                    std::vector<size_t> output_shape;
                    tensor_vector<int64_t> output_data;

                    if (input_0.m_shape.size() == 0)
                    {
                        size_t output_num_els = 1;

                        output_data = create_tensor_vector<int64_t>(output_num_els);

                        output_data[0] = input_0_data[0] - input_1_data[0];
                    }
                    else if (input_0.m_shape.size() == 1)
                    {
                        size_t output_num_els = input_0.m_shape[0];
                        output_shape = { output_num_els };

                        output_data = create_tensor_vector<int64_t>(output_num_els);

                        for (size_t i = 0; i < output_num_els; i++)
                            output_data[i] = input_0_data[i] - input_1_data[0];
                    }
                    else
                    {
                        throw std::invalid_argument(op.m_type + ": first input must be a scalar or 1D (not implemented).");
                    }

                    if (!check_output_shape(output_shape, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(output_data));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Pow")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& power = get_tensor_data(op.m_input[1], false /* make_copy */, true /* requires_float */);
                auto& output = op.m_output[0];

                if (power.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of power.");

                auto& power_data = power.get_vector<float>();

                if (power.m_shape.size() != 0 || power_data.size() != 1)
                    throw std::invalid_argument(op.m_type + ": power must be a scalar (not implemented).");

                float p = power_data[0];

                void* data = nullptr;
                size_t size = 0;
                size_t sizeof_element = 0;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();
                    size = input_data.size();

                    output.set_vector(std::move(input_data));
                    data = output.get_vector<float>().data();

                    sizeof_element = sizeof(float);
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();
                    size = input_data.size();

                    output.set_vector(std::move(input_data));
                    data = output.get_vector<uint16_t>().data();

                    sizeof_element = sizeof(uint16_t);
                }

                struct context
                {
                    void* data;
                    size_t size, threads_count;
                    float pw;

                    XnnPack* xnnpack;
                    bool error;
                };

                context prl_ctx;

                prl_ctx.data = data;
                prl_ctx.size = size;
                prl_ctx.threads_count = m_xnnpack->parallelize_threads_count();
                prl_ctx.pw = p;
                prl_ctx.xnnpack = m_xnnpack;
                prl_ctx.error = false;

                void (*pfn)(context*, size_t) = nullptr;

                if (sizeof_element == sizeof(float))
                {
                    pfn = [](context* _, size_t i) {

                        size_t start, end;
                        if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                            return;

                        float* data = (float*)_->data;
                        for (size_t j = start; j < end; j++)
                            data[j] = std::pow(data[j], _->pw);
                    };
                }
                else
                {
                    pfn = [](context* _, size_t i) {

                        float buffer[m_float32_buffer_size_w_extra_bytes];

                        size_t start, end;
                        if (!get_start_and_end(start, end, i, _->size, _->threads_count))
                            return;

                        uint16_t* data = (uint16_t*)_->data;

                        for (size_t j = start; j < end; j += m_float32_buffer_size)
                        {
                            size_t n = std::min(end - j, m_float32_buffer_size);
                            if (!_->xnnpack->convert<uint16_t, float>(&data[j], buffer, n))
                            {
                                _->error = true;
                                return;
                            }

                            for (size_t k = 0; k < n; k++)
                            {
                                float y = std::pow(buffer[k], _->pw);
                                buffer[k] = y;
                            }

                            if (!_->xnnpack->convert<float, uint16_t>(buffer, &data[j], n))
                            {
                                _->error = true;
                                return;
                            }
                        }
                    };
                }

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.threads_count);

                if (prl_ctx.error)
                    throw std::invalid_argument(op.m_type + ": conversion error between f32 and f16.");

                if (!check_output_shape(input.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Div")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0]);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (input_0.m_type == TensorDataType::float32)
                {
                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<float>();
                    auto& input_1_data = input_1.get_vector<float>();

                    auto result = m_xnnpack->divide(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input_0.m_type == TensorDataType::int64)
                {
                    if (input_0.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<int64_t>();
                    auto& input_1_data = input_1.get_vector<int64_t>();

                    auto input_0_data_fp32 = int64_to_float(input_0_data);
                    auto input_1_data_fp32 = int64_to_float(input_1_data);

                    auto result = m_xnnpack->divide(input_0.m_shape, input_0_data_fp32, input_1.m_shape, input_1_data_fp32);

                    if (input_0.m_shape.size() == 0 && input_1.m_shape.size() == 0)
                        result.first.clear();

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    auto result_i64 = float_to_int64(result.second);
                    output.set_vector(std::move(result_i64));
                }
                else
                {
                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    auto& input_1_data = input_1.get_vector<uint16_t>();

                    auto result = m_xnnpack->divide(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "MatMul")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::unspecified /* required_layout */);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                std::string cache_key;
                if (m_use_ops_cache && input_1.m_is_static_weights)
                {
                    cache_key = op.m_name;
                    if (m_first_run && m_batch_index == 0)
                    {
                        m_weights_exclusion_set.insert(input_1.m_name);
                        get_wp()->remove(input_1.m_name);
                    }
                }

                bool shape_has_leading_1 = false;
                bool first_shape_is_2d = false;

                if (input_0.m_shape.size() == 4 && input_0.m_shape[0] == 1 &&
                    input_1.m_shape.size() == 4 && input_1.m_shape[0] == 1)
                {
                    input_0.m_shape.erase(input_0.m_shape.begin());
                    input_1.m_shape.erase(input_1.m_shape.begin());
                    shape_has_leading_1 = true;
                }
                else if (input_0.m_shape.size() == 2)
                {
                    input_0.m_shape.insert(input_0.m_shape.begin(), 1);
                    if (input_1.m_shape.size() != 3)
                        first_shape_is_2d = true;
                }

                if (input_0.m_shape.size() != 3)
                    throw std::invalid_argument(op.m_type + ": shape of input 0 must have 3 dimensions (not implemented).");

                size_t n = input_0.m_shape[0];

                if (input_1.m_shape.size() == 2)
                    input_1.m_shape.insert(input_1.m_shape.begin(), n);

                if (input_1.m_shape.size() != 3)
                    throw std::invalid_argument(op.m_type + ": shape of input 1 must have 2 or 3 dimensions (not implemented).");
                else if (input_1.m_shape.size() == 3 && input_1.m_shape[0] != n)
                    throw std::invalid_argument(op.m_type + ": shape of input 1 not supported (not implemented).");

                std::vector<size_t> output_shape({ n, input_0.m_shape[1], input_1.m_shape[2] });

                if (shape_has_leading_1)
                    output_shape.insert(output_shape.begin(), 1);
                else if (first_shape_is_2d)
                    output_shape.erase(output_shape.begin());

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                if (cache_key.size() && n != 1)
                    throw std::invalid_argument(op.m_type + ": cache_key != '' and n != 1.");

                void* input_0_ptr = nullptr;
                void* input_1_ptr = nullptr;
                void* output_ptr = nullptr;
                size_t sizeof_element = 0;

                if (input_0.m_type == TensorDataType::float32)
                {
                    sizeof_element = sizeof(float);

                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                    auto& input_1_data = input_1.get_vector<float>();
                    input_1_ptr = input_1_data.data();

                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    auto& input_0_data = input_0.get_vector<float>();
                    input_0_ptr = input_0_data.data();

                    tensor_vector<float> temp = create_tensor_vector<float>(output_num_els);
                    output.set_vector(std::move(temp));
                    output_ptr = output.get_vector<float>().data();
                }
                else if (input_0.m_type == TensorDataType::float16)
                {
                    sizeof_element = sizeof(uint16_t);

                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                    auto& input_1_data = input_1.get_vector<uint16_t>();
                    input_1_ptr = input_1_data.data();

                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    input_0_ptr = input_0_data.data();

                    tensor_vector<uint16_t> temp = create_tensor_vector<uint16_t>(output_num_els);
                    output.set_vector(std::move(temp));
                    output_ptr = output.get_vector<uint16_t>().data();
                }
                else
                {
                    sizeof_element = sizeof(uint8_t);

                    if (input_1.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                    auto& input_1_data = input_1.get_vector<uint8_t>();
                    input_1_ptr = input_1_data.data();

                    if (input_0.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    auto& input_0_data = input_0.get_vector<uint8_t>();
                    input_0_ptr = input_0_data.data();

                    {
                        tensor_vector<uint8_t> temp = create_tensor_vector<uint8_t>(output_num_els);
                        output.set_vector(std::move(temp));
                        output_ptr = output.get_vector<uint8_t>().data();

                        if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                        auto s = range_to_scale(m_range_data[op.m_name]);
                        output.m_scale = s.first;
                        output.m_zero_point = s.second;
                    }
                }

                for (size_t i = 0; i < n; i++)
                {
                    std::vector<size_t> result_first;

                    if (sizeof_element == sizeof(float))
                    {
                        auto result = m_xnnpack->matrix_multiply<float, float>(
                            { input_0.m_shape[1], input_0.m_shape[2] },
                            (float*)input_0_ptr,
                            { input_1.m_shape[1], input_1.m_shape[2] },
                            (float*)input_1_ptr,
                            nullptr, nullptr,
                            (float*)output_ptr,
                            nullptr /* qu8_data */, cache_key);

                        result_first = std::move(result.first);
                    }
                    else if (sizeof_element == sizeof(uint16_t))
                    {
                        auto result = m_xnnpack->matrix_multiply<uint16_t, uint16_t>(
                            { input_0.m_shape[1], input_0.m_shape[2] },
                            (uint16_t*)input_0_ptr,
                            { input_1.m_shape[1], input_1.m_shape[2] },
                            (uint16_t*)input_1_ptr,
                            nullptr, nullptr,
                            (uint16_t*)output_ptr,
                            nullptr /* qu8_data */, cache_key);

                        result_first = std::move(result.first);
                    }
                    else
                    {
                        XnnPack::Qu8MatMulData qu8_data;

                        qu8_data.input_zero_point = input_0.m_zero_point;
                        qu8_data.input_scale = input_0.m_scale;
                        qu8_data.kernel_zero_point = input_1.m_zero_point;
                        qu8_data.kernel_scale = input_1.m_scale;
                        qu8_data.output_zero_point = output.m_zero_point;
                        qu8_data.output_scale = output.m_scale;

                        auto result = m_xnnpack->matrix_multiply<uint8_t, int32_t>(
                            { input_0.m_shape[1], input_0.m_shape[2] },
                            (uint8_t*)input_0_ptr,
                            { input_1.m_shape[1], input_1.m_shape[2] },
                            (uint8_t*)input_1_ptr,
                            nullptr, nullptr,
                            (uint8_t*)output_ptr,
                            &qu8_data, cache_key);

                        result_first = std::move(result.first);
                    }

                    if (result_first.size() != 2)
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output of matrix_multiply_fp32.");

                    input_0_ptr = (uint8_t*)input_0_ptr + input_0.m_shape[1] * input_0.m_shape[2] * sizeof_element;
                    input_1_ptr = (uint8_t*)input_1_ptr + input_1.m_shape[1] * input_1.m_shape[2] * sizeof_element;
                    output_ptr = (uint8_t*)output_ptr + result_first[0] * result_first[1] * sizeof_element;

                    if (i == n - 1)
                        push_tensor(std::move(output));
                }
            }
            else if (op.m_type == "Softmax")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::unspecified /* required_layout */);
                auto& output = op.m_output[0];

                int axis = -1;

                for (auto& a : op.m_attributes)
                    if (a.first == "axis")
                        axis = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (axis < 0)
                    axis = (int)input.m_shape.size() + axis;
                if (axis < 0 || axis >= input.m_shape.size())
                    throw std::invalid_argument(op.m_type + ": invalid axis attribute.");

                std::vector<size_t> transpose_in, transpose_out;

                if (axis != input.m_shape.size() - 1)
                {
                    for (size_t i = 0; i < input.m_shape.size(); i++)
                        if (i != axis)
                            transpose_in.push_back(i);
                    transpose_in.push_back(axis);

                    for (size_t i = 0; i < input.m_shape.size() - 1; i++)
                    {
                        if (i == axis)
                            transpose_out.push_back(input.m_shape.size() - 1);
                        transpose_out.push_back(i);
                    }
                }

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_vec = input.get_vector<float>();
                    float* input_data = input_vec.data();
                    std::vector<size_t>* input_shape = &input.m_shape;

                    std::pair<std::vector<size_t>, tensor_vector<float>> aux;

                    if (transpose_in.size())
                    {
                        aux = m_xnnpack->transpose<float>(*input_shape, input_vec, transpose_in);
                        input_data = aux.second.data();
                        input_shape = &aux.first;
                    }

                    auto result = m_xnnpack->softmax<float>(*input_shape, input_data, nullptr /* output_data_override */, nullptr /* qu8_data */, 0 /* channels_override */);

                    if (transpose_out.size())
                    {
                        aux = m_xnnpack->transpose<float>(result.first, result.second, transpose_out);
                        result = std::move(aux);
                    }

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_vec = input.get_vector<uint16_t>();
                    uint16_t* input_data = input_vec.data();
                    std::vector<size_t>* input_shape = &input.m_shape;

                    std::pair<std::vector<size_t>, tensor_vector<uint16_t>> aux;

                    if (transpose_in.size())
                    {
                        aux = m_xnnpack->transpose<uint16_t>(*input_shape, input_vec, transpose_in);
                        input_data = aux.second.data();
                        input_shape = &aux.first;
                    }

                    auto result = m_xnnpack->softmax<uint16_t>(*input_shape, input_data, nullptr /* output_data_override */, nullptr /* qu8_data */, 0 /* channels_override */);

                    if (transpose_out.size())
                    {
                        aux = m_xnnpack->transpose<uint16_t>(result.first, result.second, transpose_out);
                        result = std::move(aux);
                    }

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_vec = input.get_vector<uint8_t>();
                    uint8_t* input_data = input_vec.data();
                    std::vector<size_t>* input_shape = &input.m_shape;

                    XnnPack::Qu8SoftmaxData qu8_data;

                    qu8_data.input_scale = input.m_scale;
                    qu8_data.output_zero_point = output.m_zero_point = 0;
                    qu8_data.output_scale = output.m_scale = 0x1.0p-8f;

                    std::pair<std::vector<size_t>, tensor_vector<uint8_t>> aux;

                    if (transpose_in.size())
                    {
                        aux = m_xnnpack->transpose<uint8_t>(*input_shape, input_vec, transpose_in);
                        input_data = aux.second.data();
                        input_shape = &aux.first;
                    }

                    auto result = m_xnnpack->softmax<uint8_t>(*input_shape, input_data, nullptr /* output_data_override */, &qu8_data /* qu8_data */, 0 /* channels_override */);

                    if (transpose_out.size())
                    {
                        aux = m_xnnpack->transpose<uint8_t>(result.first, result.second, transpose_out);
                        result = std::move(aux);
                    }

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Split")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& split = get_tensor_data(op.m_input[1]);

                if (split.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of split.");

                auto& split_data = split.get_vector<int64_t>();

                void* input_ptr = nullptr;
                size_t sizeof_element = 0;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    input_ptr = input_data.data();
                    sizeof_element = sizeof(float);
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    input_ptr = input_data.data();
                    sizeof_element = sizeof(uint16_t);
                }

                int axis = -1;

                for (auto& a : op.m_attributes)
                    if (a.first == "axis")
                        axis = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (axis < 0)
                    axis = (int)input.m_shape.size() + axis;
                if (axis < 0 || axis >= input.m_shape.size())
                    throw std::invalid_argument(op.m_type + ": invalid axis attribute.");

                size_t group_size = 1;
                for (size_t i = axis + 1; i < input.m_shape.size(); i++)
                    group_size *= input.m_shape[i];

                if (op.m_output.size() != split_data.size()) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                size_t start_input_index = 0;

                for (size_t i = 0; i < split_data.size(); i++)
                {
                    int64_t dim = split_data[i];
                    auto& output = op.m_output[i];

                    if (dim <= 0)
                        throw std::invalid_argument(op.m_type + ": invalid size in split tensor.");

                    std::vector<size_t> output_shape(input.m_shape);
                    output_shape[axis] = dim;

                    size_t output_num_els = 1;
                    for (auto& s : output_shape)
                        output_num_els *= s;

                    void* output_ptr = nullptr;

                    if (sizeof_element == sizeof(float))
                    {
                        tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);
                        output.set_vector(std::move(output_data));
                        output_ptr = output.get_vector<float>().data();
                    }
                    else
                    {
                        tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);
                        output.set_vector(std::move(output_data));
                        output_ptr = output.get_vector<uint16_t>().data();
                    }

                    size_t ops_num = (output_num_els / dim) / group_size;

                    struct context
                    {
                        size_t input_index, input_stride, sizeof_element, group_size;
                        int64_t dim;
                        void* input_ptr, * output_ptr;
                    };

                    context prl_ctx;

                    prl_ctx.input_index = start_input_index;
                    prl_ctx.input_stride = input.m_shape[axis] * group_size;
                    prl_ctx.sizeof_element = sizeof_element;
                    prl_ctx.group_size = group_size;
                    prl_ctx.dim = dim;
                    prl_ctx.input_ptr = input_ptr;
                    prl_ctx.output_ptr = output_ptr;

                    void (*pfn)(context*, size_t) = [](context* _, size_t j)
                    {
                            memcpy(
                                (uint8_t*)_->output_ptr + j * _->dim * _->group_size * _->sizeof_element,
                                (uint8_t*)_->input_ptr + (_->input_index + j * _->input_stride) * _->sizeof_element,
                                _->dim * _->group_size * _->sizeof_element);
                    };

                    m_xnnpack->parallelize((void*)pfn, &prl_ctx, ops_num);

                    if (!check_output_shape(output_shape, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    push_tensor(std::move(output));

                    start_input_index += dim * group_size;
                }
            }
            else if (op.m_type == "Resize")
            {
                if (op.m_input.size() != 3 && op.m_input.size() != 4) throw std::invalid_argument(op.m_type + ": wrong number of inputs (not implemented).");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                if (op.m_input[1].m_name.size() != 0) throw std::invalid_argument(op.m_type + ": 'roi' input not supported (not implemented).");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& output = op.m_output[0];

                if (input.m_shape.size() != 4) throw std::invalid_argument(op.m_type + ": input must be 4D (not implemented).");
                if (input.m_shape[0] != 1) throw std::invalid_argument(op.m_type + ": first dimension of input's shape must be 1 (not implemented).");

                float* scales_data = nullptr;
                tensor_vector<float> scales_storage;
                std::vector<size_t> output_shape;

                if (op.m_input.size() == 3)
                {
                    auto& scales = get_tensor_data(op.m_input[2], false /* make_copy */, true /* requires_float */);

                    if (scales.m_type != TensorDataType::float32)
                        throw std::invalid_argument(op.m_type + ": wrong data type of scales.");

                    auto& scales_vector = scales.get_vector<float>();

                    if (scales_vector.size() != input.m_shape.size())
                        throw std::invalid_argument(op.m_type + ": invalid data size of scales.");
                    if (scales_vector[0] != 1 || scales_vector[1] != 1)
                        throw std::invalid_argument(op.m_type + ": first and second value of scales must be 1 (not implemented).");

                    for (size_t i = 0; i < input.m_shape.size(); i++)
                        output_shape.push_back(input.m_shape[i] * scales_vector[i]);

                    scales_data = scales_vector.data();
                }
                else if (op.m_input.size() == 4)
                {
                    auto& sizes = get_tensor_data(op.m_input[3]);

                    if (sizes.m_type != TensorDataType::int64)
                        throw std::invalid_argument(op.m_type + ": wrong data type of sizes.");

                    auto& sizes_data = sizes.get_vector<int64_t>();

                    if (sizes_data.size() != input.m_shape.size())
                        throw std::invalid_argument(op.m_type + ": invalid data size of sizes.");

                    size_t c = sizes_data.size();
                    scales_storage.resize(c);
                    for (size_t i = 0; i < c; i++)
                        scales_storage[i] = (float)sizes_data[i] / (float)input.m_shape[i];

                    if (scales_storage[0] != 1 || scales_storage[1] != 1)
                        throw std::invalid_argument(op.m_type + ": first and second value of scales (calculated from sizes) must be 1 (not implemented).");

                    for (int64_t s : sizes_data)
                        output_shape.push_back((size_t)s);

                    scales_data = scales_storage.data();
                }
                else
                {
                    throw std::invalid_argument(op.m_type + ": wrong number of inputs (not implemented).");
                }

                std::string coordinate_transformation_mode, mode, nearest_mode;
                float cubic_coeff_a = 0;

                for (auto& a : op.m_attributes)
                    if (a.first == "coordinate_transformation_mode")
                        coordinate_transformation_mode = a.second;
                    else if (a.first == "mode")
                        mode = a.second;
                    else if (a.first == "nearest_mode")
                        nearest_mode = a.second;
                    else if (a.first == "cubic_coeff_a")
                        cubic_coeff_a = std::stof(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (coordinate_transformation_mode != "asymmetric" || mode != "nearest" || nearest_mode != "floor")
                    throw std::invalid_argument(op.m_type + ": one or more attributes are not supported (not implemented).");

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                size_t sizeof_element = 0;
                void* input_data_ptr = nullptr;
                void* output_data_ptr = nullptr;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    sizeof_element = sizeof(float);
                    input_data_ptr = input_data.data();

                    tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);
                    output.set_vector(std::move(output_data));
                    output_data_ptr = output.get_vector<float>().data();
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    sizeof_element = sizeof(uint16_t);
                    input_data_ptr = input_data.data();

                    tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);
                    output.set_vector(std::move(output_data));
                    output_data_ptr = output.get_vector<uint16_t>().data();
                }
                else
                {
                    if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint8_t>();

                    sizeof_element = sizeof(uint8_t);
                    input_data_ptr = input_data.data();

                    tensor_vector<uint8_t> output_data = create_tensor_vector<uint8_t>(output_num_els);
                    output.set_vector(std::move(output_data));
                    output_data_ptr = output.get_vector<uint8_t>().data();

                    output.m_scale = input.m_scale;
                    output.m_zero_point = input.m_zero_point;
                }

                struct context
                {
                    size_t y_output_size, x_output_size, y_input_size, x_input_size, n_size, sizeof_element;
                    float y_scale, x_scale;
                    void* output_data_ptr, * input_data_ptr;
                };

                context prl_ctx;

                prl_ctx.y_output_size = output_shape[2];
                prl_ctx.x_output_size = output_shape[3];
                prl_ctx.y_scale = scales_data[2];
                prl_ctx.x_scale = scales_data[3];
                prl_ctx.y_input_size = input.m_shape[2];
                prl_ctx.x_input_size = input.m_shape[3];
                prl_ctx.n_size = input.m_shape[1];
                prl_ctx.output_data_ptr = output_data_ptr;
                prl_ctx.input_data_ptr = input_data_ptr;
                prl_ctx.sizeof_element = sizeof_element;

                void (*pfn)(context*, size_t) = [](context* _, size_t n)
                {
                    auto copy_4 = [](void* dst, void* src) { *(uint32_t*)dst = *(uint32_t*)src; };
                    auto copy_2 = [](void* dst, void* src) { *(uint16_t*)dst = *(uint16_t*)src; };
                    auto copy_1 = [](void* dst, void* src) { *(uint8_t*)dst = *(uint8_t*)src; };

                    using ptr_type = void (*)(void*, void*);
                    ptr_type copy =
                        _->sizeof_element == sizeof(float) ? (ptr_type)copy_4 :
                        _->sizeof_element == sizeof(uint16_t) ? (ptr_type)copy_2 :
                        (ptr_type)copy_1;

                    size_t pos_output = n * _->x_output_size * _->y_output_size;
                    size_t pos_input = n * _->x_input_size * _->y_input_size;

                    for (size_t y_output = 0; y_output < _->y_output_size; y_output++)
                        for (size_t x_output = 0; x_output < _->x_output_size; x_output++)
                        {
                            size_t x_input = x_output / _->x_scale;
                            size_t y_input = y_output / _->y_scale;

                            if (x_input >= _->x_input_size)
                                x_input = _->x_input_size - 1;
                            if (y_input >= _->y_input_size)
                                y_input = _->y_input_size - 1;

                            size_t final_output = pos_output + y_output * _->x_output_size + x_output;
                            size_t final_input = pos_input + y_input * _->x_input_size + x_input;

                            copy((uint8_t*)_->output_data_ptr + final_output * _->sizeof_element,
                                (uint8_t*)_->input_data_ptr + final_input * _->sizeof_element);
                        }
                };

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.n_size);

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Gather")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                int axis = 0;

                for (auto& a : op.m_attributes)
                    if (a.first == "axis")
                        axis = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& indices = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (indices.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of indices.");

                int prev_axis = axis;

                while (axis > 0 && input.m_shape.size() && input.m_shape[0] == 1)
                {
                    input.m_shape.erase(input.m_shape.begin());
                    axis--;
                }

                if (axis != 0)
                    throw std::invalid_argument(op.m_type + ": axis must be 0 (not implemented).");

                bool input_shape_1d = false;

                if (input.m_shape.size() == 1)
                {
                    input.m_shape.insert(input.m_shape.begin(), 1);
                    input_shape_1d = true;
                }

                bool indices_shape_0d = false;
                bool indices_shape_1d = false;

                if (indices.m_shape.size() == 0)
                {
                    indices.m_shape.insert(indices.m_shape.begin(), 1);
                    indices_shape_0d = true;
                }

                if (indices.m_shape.size() == 1)
                {
                    indices.m_shape.insert(indices.m_shape.begin(), 1);
                    indices_shape_1d = true;
                }

                if (indices.m_shape.size() != 2 || indices.m_shape[0] != 1)
                    throw std::invalid_argument(op.m_type + ": shape of indices must be (1,D) (not implemented).");

                std::vector<size_t> output_shape_override;
                if (input.m_shape.size() > 2)
                {
                    output_shape_override = input.m_shape;
                    output_shape_override.erase(output_shape_override.begin());
                    size_t num_els = 1;
                    for (auto& s : output_shape_override)
                        num_els *= s;
                    input.m_shape = { input.m_shape[0], num_els };
                }

                if (input.m_shape.size() != 2)
                    throw std::invalid_argument(op.m_type + ": input must be 2D or more.");

                auto& indices_data = indices.get_vector<int64_t>();

                std::vector<size_t> output_shape({ 1, indices.m_shape[1], input.m_shape[1] });

                if (input_shape_1d && indices_shape_0d)
                    output_shape.clear();
                else if (indices_shape_1d)
                    output_shape.erase(output_shape.begin());

                for (int i = 0; i < prev_axis; i++)
                    output_shape.insert(output_shape.begin(), 1);

                if (output_shape_override.size())
                    output_shape = output_shape_override;

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                size_t sizeof_element = 0;
                void* ptr_output = nullptr;
                void* ptr_input = nullptr;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);
                    output.set_vector(std::move(output_data));

                    sizeof_element = sizeof(float);
                    ptr_output = output.get_vector<float>().data();
                    ptr_input = input_data.data();
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);
                    output.set_vector(std::move(output_data));

                    sizeof_element = sizeof(uint16_t);
                    ptr_output = output.get_vector<uint16_t>().data();
                    ptr_input = input_data.data();
                }
                else
                {
                    if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<int64_t>();

                    tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);
                    output.set_vector(std::move(output_data));

                    sizeof_element = sizeof(int64_t);
                    ptr_output = output.get_vector<int64_t>().data();
                    ptr_input = input_data.data();
                }

                struct context
                {
                    std::vector<size_t>* input_shape;
                    tensor_vector<int64_t>* indices_data;
                    bool error_0, input_shape_1d;
                    size_t sizeof_element;
                    void* ptr_input, * ptr_output;
                };

                context prl_ctx;

                prl_ctx.input_shape = &input.m_shape;
                prl_ctx.indices_data = &indices_data;
                prl_ctx.error_0 = false;
                prl_ctx.input_shape_1d = input_shape_1d;
                prl_ctx.sizeof_element = sizeof_element;
                prl_ctx.ptr_input = ptr_input;
                prl_ctx.ptr_output = ptr_output;

                void (*pfn)(context*, size_t) = [](context* _, size_t i)
                {
                    int64_t index = (*_->indices_data)[i];

                    int64_t dim = (int64_t)(*_->input_shape)[_->input_shape_1d ? 1 : 0];

                    if (index < 0)
                        index = dim + index;
                    if (index < 0 || index >= dim)
                    {
                        _->error_0 = true;
                        return;
                    }

                    const size_t els = _->input_shape_1d ? 1 : (*_->input_shape)[1];

                    memcpy((uint8_t*)_->ptr_output + i * els * _->sizeof_element,
                        (uint8_t*)_->ptr_input + index * els * _->sizeof_element,
                        els * _->sizeof_element);
                };

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, indices_data.size());

                if (prl_ctx.error_0)
                    throw std::invalid_argument(op.m_type + ": invalid index in indices.");

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Slice")
            {
                if (op.m_input.size() != 3 && op.m_input.size() != 4 && op.m_input.size() != 5)
                    throw std::invalid_argument(op.m_type + ": wrong number of inputs.");

                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                auto& data = get_tensor_data(op.m_input[0]);
                auto& starts = get_tensor_data(op.m_input[1]);
                auto& ends = get_tensor_data(op.m_input[2]);
                auto* axes = op.m_input.size() > 3 ? &get_tensor_data(op.m_input[3]) : nullptr;
                auto* steps = op.m_input.size() > 4 ? &get_tensor_data(op.m_input[4]) : nullptr;
                auto& output = op.m_output[0];

                auto slice = [&m_xnnpack = m_xnnpack, &op, &starts, &ends, &axes, &steps](Tensor& data, Tensor& output, size_t current_axis) -> std::vector<size_t> {

                    if (starts.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of starts.");
                    if (ends.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of ends.");

                    if (!are_all_equal(starts.m_shape, { 1 }) && !are_all_equal(starts.m_shape, { 2 }))
                        throw std::invalid_argument(op.m_type + ": unsupported shape of starts (not implemented).");
                    if (!are_all_equal(ends.m_shape, { 1 }) && !are_all_equal(ends.m_shape, { 2 }))
                        throw std::invalid_argument(op.m_type + ": unsupported shape of ends (not implemented).");

                    auto& starts_data = starts.get_vector<int64_t>();
                    auto& ends_data = ends.get_vector<int64_t>();

                    bool last_but_one = false;

                    if (axes)
                    {
                        if (axes->m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of axes.");

                        if (!are_all_equal(axes->m_shape, { 1 }) && !are_all_equal(axes->m_shape, { 2 }))
                            throw std::invalid_argument(op.m_type + ": unsupported shape of axes (not implemented).");

                        auto& axes_data = axes->get_vector<int64_t>();

                        if (current_axis >= axes_data.size()) throw std::invalid_argument(op.m_type + ": invalid value of current_axis for axes_data.");

                        int axis = (int)axes_data[current_axis];
                        int rank = data.m_shape.size();

                        if (axis < 0)
                            axis = rank + axis;
                        if (axis < 0 || axis >= rank)
                            throw std::invalid_argument(op.m_type + ": invalid axis in axes.");

                        last_but_one = axis == rank - 2;

                        if (axis != rank - 1 && !last_but_one)
                            throw std::invalid_argument(op.m_type + ": unsupported axes value(s): slice supported on last or last but one axis only (not implemented).");
                    }

                    if (steps)
                    {
                        if (steps->m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of steps.");

                        if (!are_all_equal(steps->m_shape, { 1 }) && !are_all_equal(steps->m_shape, { 2 }))
                            throw std::invalid_argument(op.m_type + ": unsupported shape of steps (not implemented).");

                        auto& steps_data = steps->get_vector<int64_t>();

                        if (current_axis >= steps_data.size()) throw std::invalid_argument(op.m_type + ": invalid value of current_axis for steps_data.");

                        if (steps_data[current_axis] != 1)
                            throw std::invalid_argument(op.m_type + ": unsupported steps value(s) (not implemented).");
                    }

                    if (current_axis >= starts_data.size()) throw std::invalid_argument(op.m_type + ": invalid value of current_axis for starts_data.");
                    if (current_axis >= ends_data.size()) throw std::invalid_argument(op.m_type + ": invalid value of current_axis for ends_data.");

                    int64_t start = starts_data[current_axis];
                    int64_t end = ends_data[current_axis];

                    int64_t dim = (int64_t)data.m_shape[data.m_shape.size() - (last_but_one ? 2 : 1)];

                    if (start < 0)
                        start += dim;
                    if (start > dim - 1)
                        start = dim - 1;

                    if (end < 0)
                        end += dim;
                    if (end > dim)
                        end = dim;

                    if (start < 0 || start > dim || end < 0 || end > dim || start >= end)
                        throw std::invalid_argument(op.m_type + ": invalid value(s) in starts and/or ends.");

                    std::vector<size_t> output_shape(data.m_shape);

                    output_shape[output_shape.size() - (last_but_one ? 2 : 1)] = end - start;

                    size_t output_num_els = 1;
                    for (auto& s : output_shape)
                        output_num_els *= s;

                    size_t sizeof_element = 0;
                    void* ptr_data = nullptr;
                    void* ptr_output = nullptr;

                    if (data.m_type == TensorDataType::float32)
                    {
                        if (data.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of data.");

                        auto& data_data = data.get_vector<float>();

                        tensor_vector<float> temp = create_tensor_vector<float>(output_num_els);
                        output.set_vector(std::move(temp));

                        sizeof_element = sizeof(float);
                        ptr_data = data_data.data();
                        ptr_output = output.get_vector<float>().data();
                    }
                    else if (data.m_type == TensorDataType::float16)
                    {
                        if (data.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of data.");

                        auto& data_data = data.get_vector<uint16_t>();

                        tensor_vector<uint16_t> temp = create_tensor_vector<uint16_t>(output_num_els);
                        output.set_vector(std::move(temp));

                        sizeof_element = sizeof(uint16_t);
                        ptr_data = data_data.data();
                        ptr_output = output.get_vector<uint16_t>().data();
                    }
                    else
                    {
                        if (data.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of data.");

                        auto& data_data = data.get_vector<int64_t>();

                        tensor_vector<int64_t> temp = create_tensor_vector<int64_t>(output_num_els);
                        output.set_vector(std::move(temp));

                        sizeof_element = sizeof(int64_t);
                        ptr_data = data_data.data();
                        ptr_output = output.get_vector<int64_t>().data();
                    }

                    struct context
                    {
                        void* ptr_data, * ptr_output;
                        size_t output_stride, data_stride, start, sizeof_element;
                    };

                    context prl_ctx;

                    prl_ctx.start = start * (last_but_one ? data.m_shape.back() : 1);

                    prl_ctx.output_stride = output_shape.back() * (last_but_one ? output_shape[output_shape.size() - 2] : 1);
                    prl_ctx.data_stride = data.m_shape.back() * (last_but_one ? data.m_shape[data.m_shape.size() - 2] : 1);

                    size_t num = output_num_els / prl_ctx.output_stride;

                    prl_ctx.sizeof_element = sizeof_element;
                    prl_ctx.ptr_data = ptr_data;
                    prl_ctx.ptr_output = ptr_output;

                    void (*pfn)(context*, size_t) = [](context* _, size_t i)
                        {
                            memcpy((uint8_t*)_->ptr_output + _->output_stride * i * _->sizeof_element,
                                (uint8_t*)_->ptr_data + (_->data_stride * i + _->start) * _->sizeof_element,
                                _->output_stride * _->sizeof_element);
                        };

                    m_xnnpack->parallelize((void*)pfn, &prl_ctx, num);

                    return output_shape;
                };

                if (!are_all_equal(starts.m_shape, { 2 }))
                {
                    std::vector<size_t> output_shape = slice(data, output, 0);

                    if (!check_output_shape(output_shape, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
                }
                else
                {
                    Tensor interm;
                    std::vector<size_t> interm_shape = slice(data, interm, 0);
                    interm.m_shape = std::move(interm_shape);

                    std::vector<size_t> output_shape = slice(interm, output, 1);

                    if (!check_output_shape(output_shape, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "AttentionFusedOps")
            {
                if (op.m_input.size() != 4) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& q = get_tensor_data(op.m_input[0]);
                auto& k = get_tensor_data(op.m_input[1]);
                auto* s = op.m_input[2].m_name.size() ? &get_tensor_data(op.m_input[2]) : nullptr;
                auto& v = get_tensor_data(op.m_input[3]);
                auto& output = op.m_output[0];

                bool shape_has_leading_1 = false;

                if (q.m_shape.size() == 4 && q.m_shape[0] == 1 &&
                    k.m_shape.size() == 4 && k.m_shape[0] == 1 &&
                    v.m_shape.size() == 4 && v.m_shape[0] == 1)
                {
                    q.m_shape.erase(q.m_shape.begin());
                    k.m_shape.erase(k.m_shape.begin());
                    v.m_shape.erase(v.m_shape.begin());
                    shape_has_leading_1 = true;
                }

                if (q.m_shape.size() != 3 || k.m_shape.size() != 3 || v.m_shape.size() != 3)
                    throw std::invalid_argument(op.m_type + ": shapes of q, k and v must have 3 dimensions.");
                else if (q.m_shape[0] != k.m_shape[0] || q.m_shape[0] != v.m_shape[0])
                    throw std::invalid_argument(op.m_type + ": invalid shape(s) of q, k and/or v.");
                else if (s && s->m_shape.size() != 0)
                    throw std::invalid_argument(op.m_type + ": s must be a scalar.");

                size_t n = q.m_shape[0];

                std::vector<size_t> output_shape(q.m_shape);

                if (shape_has_leading_1)
                    output_shape.insert(output_shape.begin(), 1);

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                size_t sizeof_element = 0;
                void* output_ptr = nullptr;
                void* q_ptr = nullptr;
                void* k_ptr = nullptr;
                void* s_ptr = nullptr;
                void* v_ptr = nullptr;

                if (q.m_type == TensorDataType::float32)
                {
                    if (q.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of q.");
                    if (k.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of k.");
                    if (s && s->m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of s.");
                    if (v.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of v.");

                    q_ptr = q.get_vector<float>().data();
                    k_ptr = k.get_vector<float>().data();
                    s_ptr = s ? s->get_vector<float>().data() : nullptr;
                    v_ptr = v.get_vector<float>().data();

                    tensor_vector<float> temp = create_tensor_vector<float>(output_num_els);
                    output.set_vector(std::move(temp));
                    output_ptr = output.get_vector<float>().data();

                    sizeof_element = sizeof(float);
                }
                else
                {
                    if (q.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of q.");
                    if (k.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of k.");
                    if (s && s->m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of s.");
                    if (v.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of v.");

                    q_ptr = q.get_vector<uint16_t>().data();
                    k_ptr = k.get_vector<uint16_t>().data();
                    s_ptr = s ? s->get_vector<uint16_t>().data() : nullptr;
                    v_ptr = v.get_vector<uint16_t>().data();

                    tensor_vector<uint16_t> temp = create_tensor_vector<uint16_t>(output_num_els);
                    output.set_vector(std::move(temp));
                    output_ptr = output.get_vector<uint16_t>().data();

                    sizeof_element = sizeof(uint16_t);
                }

                size_t parts = m_attention_fused_ops_parts;

                if (q.m_shape[1] < parts)
                    throw std::invalid_argument(op.m_type + ": m_attention_fused_ops_parts is not valid.");

                while (q.m_shape[1] % parts)
                    ++parts;

                std::vector<size_t> aux_shape({ q.m_shape[1] / parts, k.m_shape[2] });

                size_t aux_num_els = 1;
                for (auto& s : aux_shape)
                    aux_num_els *= s;

                tensor_vector<uint8_t> aux_0 = create_tensor_vector<uint8_t>(aux_num_els * sizeof_element);
                tensor_vector<uint8_t> aux_1 = create_tensor_vector<uint8_t>(aux_num_els * sizeof_element);

                std::vector<size_t> q_part_shape({ q.m_shape[1] / parts, q.m_shape[2] });

                for (size_t i = 0; i < n; i++)
                {
                    for (size_t j = 0; j < parts; j++)
                    {
                        std::vector<size_t> result_shape;

                        if (sizeof_element == sizeof(float))
                        {
                            auto result = m_xnnpack->matrix_multiply<float, float>(
                                q_part_shape,
                                (float*)q_ptr,
                                { k.m_shape[1], k.m_shape[2] },
                                (float*)k_ptr,
                                nullptr, nullptr,
                                (float*)aux_0.data());

                            result_shape = std::move(result.first);
                        }
                        else
                        {
                            auto result = m_xnnpack->matrix_multiply<uint16_t, uint16_t>(
                                q_part_shape,
                                (uint16_t*)q_ptr,
                                { k.m_shape[1], k.m_shape[2] },
                                (uint16_t*)k_ptr,
                                nullptr, nullptr,
                                (uint16_t*)aux_0.data());

                            result_shape = std::move(result.first);
                        }

                        if (!compare_shapes(result_shape, aux_shape))
                            throw std::invalid_argument(op.m_type + ": unexpected shape of output of matrix_multiply.");

                        if (s_ptr)
                        {
                            if (sizeof_element == sizeof(float))
                            {
                                auto result = m_xnnpack->multiply<float>(
                                    aux_shape,
                                    (float*)aux_0.data(),
                                    { 1 },
                                    (float*)s_ptr,
                                    (float*)aux_1.data());

                                result_shape = std::move(result.first);
                            }
                            else
                            {
                                auto result = m_xnnpack->multiply<uint16_t>(
                                    aux_shape,
                                    (uint16_t*)aux_0.data(),
                                    { 1 },
                                    (uint16_t*)s_ptr,
                                    (uint16_t*)aux_1.data());

                                result_shape = std::move(result.first);
                            }

                            if (!compare_shapes(result_shape, aux_shape))
                                throw std::invalid_argument(op.m_type + ": unexpected shape of output of multiply.");
                        }
                        else
                        {
                            aux_0.swap(aux_1);
                        }

                        if (sizeof_element == sizeof(float))
                        {
                            auto result = m_xnnpack->softmax<float>(
                                aux_shape,
                                (float*)aux_1.data(),
                                (float*)aux_0.data());

                            result_shape = std::move(result.first);
                        }
                        else
                        {
                            auto result = m_xnnpack->softmax<uint16_t>(
                                aux_shape,
                                (uint16_t*)aux_1.data(),
                                (uint16_t*)aux_0.data());

                            result_shape = std::move(result.first);
                        }

                        if (!compare_shapes(result_shape, aux_shape))
                            throw std::invalid_argument(op.m_type + ": unexpected shape of output of softmax.");

                        if (sizeof_element == sizeof(float))
                        {
                            auto result = m_xnnpack->matrix_multiply<float, float>(
                                aux_shape,
                                (float*)aux_0.data(),
                                { v.m_shape[1], v.m_shape[2] },
                                (float*)v_ptr,
                                nullptr, nullptr,
                                (float*)output_ptr);

                            result_shape = std::move(result.first);
                        }
                        else
                        {
                            auto result = m_xnnpack->matrix_multiply<uint16_t, uint16_t>(
                                aux_shape,
                                (uint16_t*)aux_0.data(),
                                { v.m_shape[1], v.m_shape[2] },
                                (uint16_t*)v_ptr,
                                nullptr, nullptr,
                                (uint16_t*)output_ptr);

                            result_shape = std::move(result.first);
                        }

                        if (!compare_shapes(result_shape, q_part_shape))
                            throw std::invalid_argument(op.m_type + ": unexpected shape of output of matrix_multiply (2).");

                        q_ptr = (uint8_t*)q_ptr + q_part_shape[0] * q_part_shape[1] * sizeof_element;
                        output_ptr = (uint8_t*)output_ptr + q_part_shape[0] * q_part_shape[1] * sizeof_element;
                    }

                    k_ptr = (uint8_t*)k_ptr + k.m_shape[1] * k.m_shape[2] * sizeof_element;
                    v_ptr = (uint8_t*)v_ptr + v.m_shape[1] * v.m_shape[2] * sizeof_element;
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "ArgMax")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& output = op.m_output[0];

                int axis = 0;
                int keepdims = 1;
                int select_last_index = 0;

                for (auto& a : op.m_attributes)
                    if (a.first == "axis")
                        axis = std::stoi(a.second);
                    else if (a.first == "keepdims")
                        keepdims = std::stoi(a.second);
                    else if (a.first == "select_last_index")
                        select_last_index = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (axis < 0)
                    axis = (int)input.m_shape.size() + axis;
                if (axis < 0 || axis >= input.m_shape.size())
                    throw std::invalid_argument(op.m_type + ": invalid axis attribute.");

                if (axis != input.m_shape.size() - 1)
                    throw std::invalid_argument(op.m_type + ": argmax supported on last axis only (not implemented).");

                if (keepdims)
                    throw std::invalid_argument(op.m_type + ": keepdims must be 0 (not implemented).");
                if (select_last_index)
                    throw std::invalid_argument(op.m_type + ": select_last_index must be 0 (not implemented).");

                if (input.m_shape.size() != 2 || input.m_shape[0] != 1)
                    throw std::invalid_argument(op.m_type + ": shape of input must be (1,D) (not implemented).");

                std::vector<size_t> output_shape({ 1 });

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);

                int64_t& argmax = output_data[0];

                {
                    if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                    auto& input_data = input.get_vector<int64_t>();

                    int64_t max = std::numeric_limits<int64_t>::min();

                    argmax = 0;
                    for (int64_t i = 0; i < input_data.size(); i++)
                    {
                        int64_t& v = input_data[i];
                        if (v > max)
                        {
                            max = v;
                            argmax = i;
                        }
                    }
                }

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(output_data));
                push_tensor(std::move(output));
            }
            else if (op.m_type == "Shape")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& output = op.m_output[0];

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                if (input.m_shape.size() == 0)
                    throw std::invalid_argument(op.m_type + ": shape of input not available.");

                std::vector<size_t> output_shape({ input.m_shape.size() });

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);

                for (size_t i = 0; i < output_num_els; i++)
                    output_data[i] = input.m_shape[i];

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(output_data));
                push_tensor(std::move(output));
            }
            else if (op.m_type == "Where")
            {
                if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0]);
                auto& input_1 = get_tensor_data(op.m_input[1]);
                auto& input_2 = get_tensor_data(op.m_input[2]);
                auto& output = op.m_output[0];

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                bool input_0_scalar = input_0.m_shape.size() == 0;
                bool input_1_scalar = input_1.m_shape.size() == 0;
                bool input_2_scalar = input_2.m_shape.size() == 0;

                if (input_0_scalar)
                    throw std::invalid_argument(op.m_type + ": condition cannot be a scalar (not implemented).");

                if ((!input_1_scalar && !compare_shapes(input_0.m_shape, input_1.m_shape)) ||
                    (!input_2_scalar && !compare_shapes(input_0.m_shape, input_2.m_shape)))
                    throw std::invalid_argument(op.m_type + ": shapes of condition, A and/or B not equal (broadcasting not implemented).");

                if (input_0.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of condition (not implemented).");

                auto& input_0_data = input_0.get_vector<int64_t>();

                std::vector<size_t> output_shape(input_0.m_shape);

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                if (input_1.m_type == TensorDataType::int64 || input_2.m_type == TensorDataType::int64)
                {
                    tensor_vector<int64_t>* input_1_data = nullptr;
                    tensor_vector<int64_t>* input_2_data = nullptr;
                    tensor_vector<int64_t> temp_1, temp_2;

                    if (input_1.m_type == TensorDataType::int64)
                    {
                        input_1_data = &input_1.get_vector<int64_t>();
                    }
                    else if (input_1.m_type == TensorDataType::float32)
                    {
                        temp_1 = float_to_int64(input_1.get_vector<float>());
                        input_1_data = &temp_1;
                    }
                    else if (input_1.m_type == TensorDataType::float16)
                    {
                        auto temp = m_xnnpack->convert<uint16_t, float>(input_1.get_vector<uint16_t>());
                        temp_1 = float_to_int64(temp);
                        input_1_data = &temp_1;
                    }

                    if (input_2.m_type == TensorDataType::int64)
                    {
                        input_2_data = &input_2.get_vector<int64_t>();
                    }
                    else if (input_2.m_type == TensorDataType::float32)
                    {
                        temp_2 = float_to_int64(input_2.get_vector<float>());
                        input_2_data = &temp_2;
                    }
                    else if (input_2.m_type == TensorDataType::float16)
                    {
                        auto temp = m_xnnpack->convert<uint16_t, float>(input_2.get_vector<uint16_t>());
                        temp_2 = float_to_int64(temp);
                        input_2_data = &temp_2;
                    }

                    if (!input_1_data) throw std::invalid_argument(op.m_type + ": wrong data type of A (not implemented).");
                    if (!input_2_data) throw std::invalid_argument(op.m_type + ": wrong data type of B (not implemented).");

                    tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);

                    for (size_t i = 0; i < output_num_els; i++)
                        output_data[i] = input_0_data[i] ? (*input_1_data)[input_1_scalar ? 0 : i] : (*input_2_data)[input_2_scalar ? 0 : i];

                    output.set_vector(std::move(output_data));
                }
                else if (input_1.m_type == TensorDataType::float32)
                {
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of A (not implemented).");
                    if (input_2.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of B (not implemented).");

                    auto& input_1_data = input_1.get_vector<float>();
                    auto& input_2_data = input_2.get_vector<float>();

                    tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);

                    for (size_t i = 0; i < output_num_els; i++)
                        output_data[i] = input_0_data[i] ? input_1_data[input_1_scalar ? 0 : i] : input_2_data[input_2_scalar ? 0 : i];

                    output.set_vector(std::move(output_data));
                }
                else if (input_1.m_type == TensorDataType::float16)
                {
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of A (not implemented).");
                    if (input_2.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of B (not implemented).");

                    auto& input_1_data = input_1.get_vector<uint16_t>();
                    auto& input_2_data = input_2.get_vector<uint16_t>();

                    tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);

                    for (size_t i = 0; i < output_num_els; i++)
                        output_data[i] = input_0_data[i] ? input_1_data[input_1_scalar ? 0 : i] : input_2_data[input_2_scalar ? 0 : i];

                    output.set_vector(std::move(output_data));
                }
                else
                    throw std::invalid_argument(op.m_type + ": wrong data type of A and/or B (not implemented).");

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Expand")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& shape = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                if (shape.m_shape.size() != 1)
                    throw std::invalid_argument(op.m_type + ": shape must be 1D.");
                if (input.m_shape.size() == 0)
                    throw std::invalid_argument(op.m_type + ": invalid shape of input.");

                while (input.m_shape.size() < shape.m_shape[0])
                    input.m_shape.insert(input.m_shape.begin(), 1);

                if (input.m_shape.size() != shape.m_shape[0])
                    throw std::invalid_argument(op.m_type + ": shape of input not matching 'shape'.");

                if (shape.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of shape.");

                auto& shape_data = shape.get_vector<int64_t>();

                size_t last_dim = -1;

                for (size_t i = 0; i < shape_data.size(); i++)
                {
                    int64_t dimi = (int64_t)input.m_shape[i];
                    int64_t dimo = shape_data[i];

                    if (dimi <= 0 || dimo <= 0)
                        throw std::invalid_argument(op.m_type + ": dimension <= 0.");
                    if (dimi > dimo)
                        throw std::invalid_argument(op.m_type + ": input dimension > output dimension.");
                    if (dimi != dimo)
                    {
                        if (dimi != 1)
                            throw std::invalid_argument(op.m_type + ": invalid input dimension.");
                        last_dim = i;
                    }
                }

                std::vector<size_t> output_shape;
                for (auto& d : shape_data)
                    output_shape.push_back((size_t)d);

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                size_t sizeof_element = 0;
                void* input_data = nullptr;
                void* output_data = nullptr;
                size_t input_data_size = 0;

                if (input.m_type == TensorDataType::int64)
                {
                    if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                    sizeof_element = sizeof(int64_t);

                    auto& input_vec = input.get_vector<int64_t>();
                    input_data = input_vec.data();
                    input_data_size = input_vec.size();

                    tensor_vector<int64_t> output_vec = create_tensor_vector<int64_t>(output_num_els);
                    output.set_vector(std::move(output_vec));
                    output_data = output.get_vector<int64_t>().data();
                }
                else if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                    sizeof_element = sizeof(float);

                    auto& input_vec = input.get_vector<float>();
                    input_data = input_vec.data();
                    input_data_size = input_vec.size();

                    tensor_vector<float> output_vec = create_tensor_vector<float>(output_num_els);
                    output.set_vector(std::move(output_vec));
                    output_data = output.get_vector<float>().data();
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                    sizeof_element = sizeof(uint16_t);

                    auto& input_vec = input.get_vector<uint16_t>();
                    input_data = input_vec.data();
                    input_data_size = input_vec.size();

                    tensor_vector<uint16_t> output_vec = create_tensor_vector<uint16_t>(output_num_els);
                    output.set_vector(std::move(output_vec));
                    output_data = output.get_vector<uint16_t>().data();
                }

                if (last_dim == -1)
                {
                    if (input_data_size != output_num_els)
                        throw std::invalid_argument(op.m_type + ": input and output vector size mismatch.");

                    memcpy(output_data, input_data, input_data_size * sizeof_element);
                }
                else
                {
                    struct context
                    {
                        size_t num_elements, sizeof_element, pos_src, pos_dst, num_copies;
                        void* input_data, * output_data;
                    };

                    std::vector<context> work_items;

                    std::vector<size_t> s = output_shape;
                    size_t pos_dst = 0;

                    size_t num_elements = 1;
                    for (size_t i = last_dim + 1; i < input.m_shape.size(); i++)
                        num_elements *= input.m_shape[i];

                    std::function<void(size_t, size_t)> expand = [&](size_t dim, size_t val)
                    {
                        if (dim != -1) s[dim] = val;
                        dim++;

                        if (dim < last_dim)
                        {
                            for (size_t i = 0; i < output_shape[dim]; i++)
                                expand(dim, i);
                        }
                        else
                        {
                            size_t num_copies = output_shape[dim];

                            size_t pos_src = 0;
                            for (size_t i = 0; i <= last_dim; i++)
                            {
                                size_t m = s[i];
                                if (m && m < input.m_shape[i])
                                {
                                    for (size_t j = i + 1; j < input.m_shape.size(); j++)
                                        m *= input.m_shape[j];

                                    pos_src += m;
                                }
                            }

                            context prl_ctx;

                            prl_ctx.num_elements = num_elements;
                            prl_ctx.sizeof_element = sizeof_element;
                            prl_ctx.pos_src = pos_src;
                            prl_ctx.pos_dst = pos_dst;
                            prl_ctx.input_data = input_data;
                            prl_ctx.output_data = output_data;
                            prl_ctx.num_copies = num_copies;

                            work_items.push_back(prl_ctx);

                            pos_dst += num_elements * num_copies;
                        }
                    };

                    expand(-1, 0);

                    if (pos_dst != output_num_els)
                        throw std::invalid_argument(op.m_type + ": pointer error.");

                    void (*pfn)(context*, size_t) = nullptr;

                    pfn = [](context* _, size_t x)
                    {
                        _ += x;

                        size_t n = _->num_elements * _->sizeof_element;

                        for (size_t i = 0; i < _->num_copies; i++)
                        {
                            memcpy((uint8_t*)_->output_data + _->pos_dst * _->sizeof_element + n * i,
                                (uint8_t*)_->input_data + _->pos_src * _->sizeof_element,
                                n);
                        }
                    };

                    m_xnnpack->parallelize((void*)pfn, work_items.data(), work_items.size());
                }

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Cast")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], false /* make_copy */, true /* requires_float */);
                auto& output = op.m_output[0];

                int to = -1;

                for (auto& a : op.m_attributes)
                    if (a.first == "to")
                    {
                        to = std::stoi(a.second);
                    }
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                if (to == -1)
                    throw std::invalid_argument(op.m_type + ": 'to' attribute not found.");

                std::vector<size_t> output_shape(input.m_shape);

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                if (to == 1 /* FLOAT */)
                {
                    if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                    auto& input_data = input.get_vector<int64_t>();

                    tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);

                    for (size_t i = 0; i < output_num_els; i++)
                        output_data[i] = (float)input_data[i];

                    output.set_vector(std::move(output_data));
                }
                else if (to == 9 /* BOOL */ || to == 7 /* INT64 */ || to == 6 /* INT32 */)
                {
                    tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);

                    if (input.m_type == TensorDataType::float32)
                    {
                        if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                        auto& input_data = input.get_vector<float>();

                        for (size_t i = 0; i < output_num_els; i++)
                            output_data[i] = (int64_t)input_data[i];
                    }
                    else
                    {
                        if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input (not implemented).");

                        auto& input_data = input.get_vector<int64_t>();

                        for (size_t i = 0; i < output_num_els; i++)
                            output_data[i] = input_data[i];
                    }

                    output.set_vector(std::move(output_data));
                }
                else
                    throw std::invalid_argument(op.m_type + ": requested cast not implemented.");

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Squeeze")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& data = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& axes = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                if (axes.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of axes.");

                auto& axes_data = axes.get_vector<int64_t>();

                if (axes_data.size() == 0)
                    throw std::invalid_argument(op.m_type + ": axes cannot be empty (not implemented).");

                int rank = data.m_shape.size();

                for (auto& a : axes_data)
                {
                    if (a < 0)
                        a = rank + a;
                    if (a < 0 || a >= rank)
                        throw std::invalid_argument(op.m_type + ": wrong data in axes.");
                }

                std::sort(axes_data.begin(), axes_data.end(), std::greater<>());

                int64_t prev = -1;
                for (auto& a : axes_data)
                {
                    if (a == prev)
                        throw std::invalid_argument(op.m_type + ": duplicate value in axes.");
                    else
                        prev = a;

                    if (a >= data.m_shape.size())
                        throw std::invalid_argument(op.m_type + ": wrong data in axes.");

                    data.m_shape.erase(data.m_shape.begin() + a);
                }

                if (!check_output_shape(data.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.m_type = data.m_type;
                output.m_data = std::move(data.m_data);

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Neg")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& output = op.m_output[0];

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    std::vector<size_t> minus_1_shape({ 1 });
                    tensor_vector<float> minus_1_data = create_tensor_vector<float>(1);
                    minus_1_data[0] = -1;

                    auto result = m_xnnpack->multiply(input.m_shape, input_data.data(), minus_1_shape, minus_1_data.data());

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input.m_type == TensorDataType::int64)
                {
                    if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");

                    auto& input_data = input.get_vector<int64_t>();

                    std::vector<size_t> minus_1_shape({ 1 });
                    tensor_vector<float> minus_1_data = create_tensor_vector<float>(1);
                    minus_1_data[0] = -1;

                    auto input_data_fp32 = int64_to_float(input_data);

                    auto result = m_xnnpack->multiply(input.m_shape, input_data_fp32.data(), minus_1_shape, minus_1_data.data());

                    if (input.m_shape.size() == 0)
                        result.first.clear();

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    auto result_i64 = float_to_int64(result.second);
                    output.set_vector(std::move(result_i64));
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    std::vector<size_t> minus_1_shape({ 1 });
                    tensor_vector<uint16_t> minus_1_data = create_tensor_vector<uint16_t>(1);
                    minus_1_data[0] = 0b1011110000000000;

                    auto result = m_xnnpack->multiply(input.m_shape, input_data.data(), minus_1_shape, minus_1_data.data());

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "ConstantOfShape")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& output = op.m_output[0];

                std::string value;

                for (auto& a : op.m_attributes)
                    if (a.first == "value")
                        value = a.second;
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                if (value.size() == 0)
                    throw std::invalid_argument(op.m_type + ": 'value' attribute not specified (not implemented).");

                if (input.m_shape.size() != 1)
                    throw std::invalid_argument(op.m_type + ": input must be 1D.");

                if (input.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                auto& input_data = input.get_vector<int64_t>();

                std::vector<size_t> output_shape;
                for (auto& d : input_data)
                    output_shape.push_back((size_t)d);

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);

                float v = std::stof(value);
                for (size_t i = 0; i < output_num_els; i++)
                    output_data[i] = v;

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(output_data));
                push_tensor(std::move(output));
            }
            else if (op.m_type == "Range")
            {
                if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& start = get_tensor_data(op.m_input[0]);
                auto& limit = get_tensor_data(op.m_input[1]);
                auto& delta = get_tensor_data(op.m_input[2]);
                auto& output = op.m_output[0];

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                if (start.m_shape.size() != 0) throw std::invalid_argument(op.m_type + ": start must be a scalar.");
                if (limit.m_shape.size() != 0) throw std::invalid_argument(op.m_type + ": limit must be a scalar.");
                if (delta.m_shape.size() != 0) throw std::invalid_argument(op.m_type + ": delta must be a scalar.");

                if (start.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of start (not implemented).");
                if (limit.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of limit (not implemented).");
                if (delta.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of delta (not implemented).");

                auto& start_data = start.get_vector<int64_t>();
                auto& limit_data = limit.get_vector<int64_t>();
                auto& delta_data = delta.get_vector<int64_t>();

                int64_t s = start_data[0];
                int64_t l = limit_data[0];
                int64_t d = delta_data[0];

                if (d != 1)
                    throw std::invalid_argument(op.m_type + ": delta must be 1 (not implemented).");
                if (s >= l)
                    throw std::invalid_argument(op.m_type + ": start must be less than limit.");

                size_t output_num_els = l - s;
                std::vector<size_t> output_shape({ output_num_els });

                tensor_vector<int64_t> output_data = create_tensor_vector<int64_t>(output_num_els);

                for (size_t i = 0; i < output_num_els; i++)
                    output_data[i] = s + i;

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(output_data));
                push_tensor(std::move(output));
            }
            else if (op.m_type == "Less" ||
                op.m_type == "Greater" ||
                op.m_type == "Equal" ||
                op.m_type == "And")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input_0 = get_tensor_data(op.m_input[0], false /* make_copy */, true /* requires_float */);
                auto& input_1 = get_tensor_data(op.m_input[1], false /* make_copy */, true /* requires_float */);
                auto& output = op.m_output[0];

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                bool (*pcompare)(int64_t, int64_t) = nullptr;

                if (op.m_type == "Less")
                {
                    pcompare = [](int64_t a, int64_t b) -> bool { return a < b; };
                }
                else if (op.m_type == "Greater")
                {
                    pcompare = [](int64_t a, int64_t b) -> bool { return a > b; };
                }
                else if (op.m_type == "Equal")
                {
                    pcompare = [](int64_t a, int64_t b) -> bool { return a == b; };
                }
                else if (op.m_type == "And")
                {
                    pcompare = [](int64_t a, int64_t b) -> bool { return a && b; };
                }
                else
                    throw std::invalid_argument(op.m_type + ": unrecognized operation.");

                size_t p;
                for (p = 0; p < input_1.m_shape.size(); p++)
                    if (input_1.m_shape[p] != 1)
                        break;
                size_t input_1_mod =
                    p == input_1.m_shape.size() ? 1 :
                    p == input_1.m_shape.size() - 1 && input_1.m_shape.size() == input_0.m_shape.size() && input_1.m_shape[p] == input_0.m_shape[p] ? input_1.m_shape[p] :
                    0;

                float fixed_point_const =
                    input_0.m_type == TensorDataType::float32 && input_1.m_type == TensorDataType::float32 ? 10000 : 1;

                auto get_input_0_data = [&](size_t i) -> int64_t
                {
                    if (input_0.m_type == TensorDataType::int64)
                    {
                        if (input_0.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of A (not implemented).");
                        auto& input_0_data = input_0.get_vector<int64_t>();
                        return input_0_data[i];
                    }
                    else
                    {
                        if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of A (not implemented).");
                        auto& input_0_data = input_0.get_vector<float>();
                        return (int64_t)(input_0_data[i] * fixed_point_const);
                    }
                };

                auto get_input_1_data = [&](size_t i) -> int64_t
                {
                    if (input_1_mod) i = i % input_1_mod;

                    if (input_1.m_type == TensorDataType::int64)
                    {
                        if (input_1.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of B (not implemented).");
                        auto& input_1_data = input_1.get_vector<int64_t>();
                        return input_1_data[i];
                    }
                    else
                    {
                        if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of B (not implemented).");
                        auto& input_1_data = input_1.get_vector<float>();
                        return (int64_t)(input_1_data[i] * fixed_point_const);
                    }
                };

                tensor_vector<int64_t> output_data;
                std::vector<size_t> output_shape;

                if (input_1_mod ||
                    compare_shapes(input_0.m_shape, input_1.m_shape))
                {
                    output_shape = input_1_mod ? input_1.m_shape : input_0.m_shape;

                    size_t output_num_els = 1;
                    for (auto& s : output_shape)
                        output_num_els *= s;

                    output_data = create_tensor_vector<int64_t>(output_num_els);

                    for (size_t i = 0; i < output_num_els; i++)
                        output_data[i] = pcompare(get_input_0_data(i), get_input_1_data(i)) ? 1 : 0;
                }
                else if (input_0.m_shape.size() == 1 &&
                    input_1.m_shape.size() == 2 &&
                    input_1.m_shape[1] == 1)
                {
                    size_t w = input_0.m_shape[0];
                    size_t h = input_1.m_shape[0];

                    output_shape = { h, w };

                    size_t output_num_els = 1;
                    for (auto& s : output_shape)
                        output_num_els *= s;

                    output_data = create_tensor_vector<int64_t>(output_num_els);

                    size_t i = 0;
                    for (size_t y = 0; y < h; y++)
                        for (size_t x = 0; x < w; x++)
                            output_data[i++] = pcompare(get_input_0_data(x), get_input_1_data(y)) ? 1 : 0;
                }
                else
                {
                    throw std::invalid_argument(op.m_type + ": shapes of A and B not compatible (not implemented).");
                }

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(output_data));
                push_tensor(std::move(output));
            }
            else if (op.m_type == "ScaledDotProductAttention")
            {
                if (op.m_input.size() != 5 && op.m_input.size() != 6) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& q = get_tensor_data(op.m_input[0]);
                auto& k = get_tensor_data(op.m_input[1]);
                auto& s = get_tensor_data(op.m_input[2]);
                auto& m = get_tensor_data(op.m_input[3]);
                auto& v = get_tensor_data(op.m_input[4]);
                auto* s2 = op.m_input.size() == 6 ? &get_tensor_data(op.m_input[5]) : nullptr;
                auto& output = op.m_output[0];

                if (q.m_shape.size() != 4) throw std::invalid_argument(op.m_type + ": invalid shape of query.");
                if (k.m_shape.size() != 4) throw std::invalid_argument(op.m_type + ": invalid shape of key.");
                if (v.m_shape.size() != 4) throw std::invalid_argument(op.m_type + ": invalid shape of value.");

                if (s.m_shape.size() != 0 && !(s.m_shape.size() == 1 && s.m_shape[0] == 1))
                    throw std::invalid_argument(op.m_type + ": invalid shape of scale.");
                if (s2 && s2->m_shape.size() != 0 && !(s2->m_shape.size() == 1 && s2->m_shape[0] == 1))
                    throw std::invalid_argument(op.m_type + ": invalid shape of second scale.");
                if (m.m_shape.size() != 2 && !(m.m_shape.size() == 4 && m.m_shape[0] == 1 && m.m_shape[1] == 1))
                    throw std::invalid_argument(op.m_type + ": invalid shape of mask.");

                size_t batch_size = q.m_shape[0];
                size_t query_heads = q.m_shape[1];
                size_t query_tokens = q.m_shape[2];
                size_t query_key_channels = q.m_shape[3];
                size_t key_value_heads = k.m_shape[1];
                size_t key_value_tokens = k.m_shape[2];
                size_t value_channels = v.m_shape[3];

                if (q.m_type == TensorDataType::float32)
                {
                    if (q.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of query.");
                    if (k.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of key.");
                    if (v.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of value.");
                    if (s.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of scale.");
                    if (m.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of mask.");
                    if (s2 && s2->m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of second scale.");

                    auto& q_data = q.get_vector<float>();
                    auto& k_data = k.get_vector<float>();
                    auto& v_data = v.get_vector<float>();
                    auto& s_data = s.get_vector<float>();
                    auto& m_data = m.get_vector<float>();
                    auto* s2_data = s2 ? &s2->get_vector<float>() : nullptr;

                    auto scale = tensor_vector<float>(query_key_channels, s2_data ? s2_data->data()[0] * s_data[0] : 1.0f / s_data[0]);

                    auto result = m_xnnpack->scaled_dot_product_attention<float>(
                        q_data, k_data, v_data,
                        scale, m_data,
                        batch_size, query_heads, query_tokens,
                        key_value_heads, key_value_tokens,
                        query_key_channels, value_channels);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (q.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of query.");
                    if (k.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of key.");
                    if (v.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of value.");
                    if (s.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of scale.");
                    if (m.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of mask.");
                    if (s2 && s2->m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of second scale.");

                    auto& q_data = q.get_vector<uint16_t>();
                    auto& k_data = k.get_vector<uint16_t>();
                    auto& v_data = v.get_vector<uint16_t>();
                    auto& s_data = s.get_vector<uint16_t>();
                    auto& m_data = m.get_vector<uint16_t>();
                    auto* s2_data = s2 ? &s2->get_vector<uint16_t>() : nullptr;

                    float val = 0;
                    if (!m_xnnpack->convert<uint16_t, float>(s_data.data(), &val, 1))
                        throw std::invalid_argument(op.m_type + ": fp16 to fp32 conversion error.");

                    if (s2_data)
                    {
                        float val2 = 0;
                        if (!m_xnnpack->convert<uint16_t, float>(s2_data->data(), &val2, 1))
                            throw std::invalid_argument(op.m_type + ": fp16 to fp32 conversion error.");

                        val = val2 * val;
                    }
                    else
                    {
                        val = 1.0f / val;
                    }

                    uint16_t val16 = 0;
                    if (!m_xnnpack->convert<float, uint16_t>(&val, &val16, 1))
                        throw std::invalid_argument(op.m_type + ": fp32 to fp16 conversion error.");

                    auto scale = tensor_vector<uint16_t>(query_key_channels, val16);

                    auto result = m_xnnpack->scaled_dot_product_attention<uint16_t>(
                        q_data, k_data, v_data,
                        scale, m_data,
                        batch_size, query_heads, query_tokens,
                        key_value_heads, key_value_tokens,
                        query_key_channels, value_channels);

                    if (!check_output_shape(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Trilu")
            {
                if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0]);
                auto& k_tensor = get_tensor_data(op.m_input[1]);
                auto& output = op.m_output[0];

                std::string upper = "1";

                for (auto& a : op.m_attributes)
                    if (a.first == "upper")
                        upper = a.second;
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                if (upper != "1")
                    throw std::invalid_argument(op.m_type + ": 'upper' must be 1 (not implemented).");

                if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");
                if (k_tensor.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of k.");

                auto& input_data = input.get_vector<float>();
                auto& k_data = k_tensor.get_vector<int64_t>();

                if (input.m_shape.size() != 2)
                    throw std::invalid_argument(op.m_type + ": input must be 2D (not implemented).");
                if (k_tensor.m_shape.size() != 0)
                    throw std::invalid_argument(op.m_type + ": second input (k) must be a scalar (not implemented).");

                std::vector<size_t> output_shape = input.m_shape;

                size_t output_num_els = 1;
                for (auto& s : output_shape)
                    output_num_els *= s;

                tensor_vector<float> output_data = create_tensor_vector<float>(output_num_els);

                int w = (int)input.m_shape[1];
                int h = (int)input.m_shape[0];
                int k = (int)k_data[0];

                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int index = y * w + x;
                        output_data[index] = x - k >= y ? input_data[index] : 0;
                    }

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(output_data));
                push_tensor(std::move(output));
            }
            else if (op.m_type == "ScatterND")
            {
                if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& indices = get_tensor_data(op.m_input[1]);
                auto& updates = get_tensor_data(op.m_input[2]);
                auto& output = op.m_output[0];

                if (op.m_attributes.size())
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

                size_t rank = input.m_shape.size();

                if (!rank ||
                    indices.m_shape.size() != rank + 1 ||
                    updates.m_shape.size() != rank ||
                    indices.m_shape[rank] != rank)
                {
                    throw std::invalid_argument(op.m_type + ": invalid shape of one or more inputs.");
                }

                if (indices.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of indices.");

                auto& indices_data = indices.get_vector<int64_t>();

                size_t sizeof_element = 0;
                void* input_data_data = nullptr;
                size_t input_data_size = 0;
                void* updates_data_data = nullptr;
                size_t updates_data_size = 0;

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");
                    if (updates.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of updates.");

                    auto& input_data = input.get_vector<float>();
                    input_data_size = input_data.size();

                    output.set_vector(std::move(input_data));
                    input_data_data = output.get_vector<float>().data();

                    sizeof_element = sizeof(float);

                    auto& updates_data = updates.get_vector<float>();
                    updates_data_data = updates_data.data();
                    updates_data_size = updates_data.size();
                }
                else
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");
                    if (updates.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of updates.");

                    auto& input_data = input.get_vector<uint16_t>();
                    input_data_size = input_data.size();

                    output.set_vector(std::move(input_data));
                    input_data_data = output.get_vector<uint16_t>().data();

                    sizeof_element = sizeof(uint16_t);

                    auto& updates_data = updates.get_vector<uint16_t>();
                    updates_data_data = updates_data.data();
                    updates_data_size = updates_data.size();
                }

                if (updates_data_size * rank != indices_data.size())
                    throw std::invalid_argument(op.m_type + ": sizes of updates and indices not compatible.");

                std::vector<size_t> dims(rank);
                for (size_t i = 0; i < rank; i++)
                {
                    size_t v = 1;
                    for (size_t j = i + 1; j < rank; j++)
                        v *= input.m_shape[j];
                    dims[i] = v;
                }

                auto copy_4 = [](void* dst, void* src) { *(uint32_t*)dst = *(uint32_t*)src; };
                auto copy_2 = [](void* dst, void* src) { *(uint16_t*)dst = *(uint16_t*)src; };

                using ptr_type = void (*)(void*, void*);
                ptr_type copy =
                    sizeof_element == sizeof(float) ? (ptr_type)copy_4 :
                    sizeof_element == sizeof(uint16_t) ? (ptr_type)copy_2 :
                    nullptr;

                struct context
                {
                    bool error;
                    ptr_type copy;
                    size_t rank, * dims, sizeof_element, input_data_size;
                    int64_t* indices_data;
                    void* input_data_data, * updates_data_data;
                };

                context prl_ctx;

                prl_ctx.error = false;
                prl_ctx.copy = copy;
                prl_ctx.rank = rank;
                prl_ctx.dims = dims.data();
                prl_ctx.indices_data = indices_data.data();
                prl_ctx.input_data_data = input_data_data;
                prl_ctx.sizeof_element = sizeof_element;
                prl_ctx.input_data_size = input_data_size;
                prl_ctx.updates_data_data = updates_data_data;

                void (*pfn)(context*, size_t) = [](context* _, size_t i)
                {
                    int64_t* index = &_->indices_data[i * _->rank];

                    size_t pos = 0;
                    for (size_t j = 0; j < _->rank; j++)
                        pos += index[j] * _->dims[j];
                    if (pos >= _->input_data_size)
                    {
                        _->error = true;
                        return;
                    }

                    _->copy((uint8_t*)_->input_data_data + pos * _->sizeof_element, (uint8_t*)_->updates_data_data + i * _->sizeof_element);
                };

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, updates_data_size);

                if (prl_ctx.error)
                    throw std::invalid_argument(op.m_type + ": invalid index in indices.");

                if (!check_output_shape(input.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));
            }
            else if (op.m_type == "MaxPool")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                std::vector<int> dilations, kernel_shape, pads, strides;
                int ceil_mode = 0;

                for (auto& a : op.m_attributes)
                    if (a.first == "dilations")
                        dilations = string_to_int_vec<int>(a.second);
                    else if (a.first == "ceil_mode")
                        ceil_mode = std::stoi(a.second);
                    else if (a.first == "kernel_shape")
                        kernel_shape = string_to_int_vec<int>(a.second);
                    else if (a.first == "pads")
                        pads = string_to_int_vec<int>(a.second);
                    else if (a.first == "strides")
                        strides = string_to_int_vec<int>(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (!are_all_equal(dilations, { 1, 1 }))
                    throw std::invalid_argument(op.m_type + ": invalid dilations attribute value (not implemented).");
                if (ceil_mode != 0)
                    throw std::invalid_argument(op.m_type + ": invalid ceil_mode attribute value (not implemented).");

                auto& x = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::nhwc /* required_layout */);
                auto& output = op.m_output[0];

                if (x.m_layout != TensorDataLayout::nhwc) throw std::invalid_argument(op.m_type + ": wrong layout of X.");

                std::vector<size_t> result_first;

                if (x.m_type == TensorDataType::float32)
                {
                    if (x.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of X.");

                    auto& x_data = x.get_vector<float>();

                    auto result = m_xnnpack->maxpool_nhwc<float>(
                        x.m_shape, x_data,
                        dilations, kernel_shape, pads, strides);

                    result_first = std::move(result.first);

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (x.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of X.");

                    auto& x_data = x.get_vector<uint16_t>();

                    auto result = m_xnnpack->maxpool_nhwc<uint16_t>(
                        x.m_shape, x_data,
                        dilations, kernel_shape, pads, strides);

                    result_first = std::move(result.first);

                    output.set_vector(std::move(result.second));
                }

                if (result_first.size() != 4 ||
                    !check_output_shape({ result_first[0], result_first[3], result_first[1], result_first[2] }, output.m_shape))
                {
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
                }

                output.m_layout = TensorDataLayout::nhwc;
                output.m_shape = std::move(result_first);

                push_tensor(std::move(output));
            }
            else if (op.m_type == "Flatten")
            {
                if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
                if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

                auto& input = get_tensor_data(op.m_input[0], true /* make_copy */);
                auto& output = op.m_output[0];

                int axis = 1;

                for (auto& a : op.m_attributes)
                    if (a.first == "axis")
                        axis = std::stoi(a.second);
                    else
                        throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

                if (axis < 0)
                    axis = (int)input.m_shape.size() + axis;
                if (axis < 0 || axis >= input.m_shape.size())
                    throw std::invalid_argument(op.m_type + ": invalid axis attribute.");

                size_t height = 1, width = 1;
                for (size_t i = 0; i < input.m_shape.size(); i++)
                    if (i < axis)
                        height *= input.m_shape[i];
                    else
                        width *= input.m_shape[i];

                std::vector<size_t> output_shape = { height, width };

                if (!check_output_shape(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.m_type = input.m_type;
                output.m_data = std::move(input.m_data);

                output.m_scale = input.m_scale;
                output.m_zero_point = input.m_zero_point;

                push_tensor(std::move(output));
            }
            else
                throw std::invalid_argument("Operator not implemented: " + op.m_type + ".");
        }

        m_batch_size = 1;
        m_batch_index = 0;
        m_batch_cache.clear();
        if (free_ops_cache) m_xnnpack->free_ops_cache();

        if (m_ops_times_printf)
        {
            std::chrono::duration<double, std::milli> t = std::chrono::high_resolution_clock::now() - hrc_now;

            static std::map<std::string, double> m0;

            m0[op.m_type] += t.count();

            if (m_ops_queue.size() == 1)
            {
                printf("\033[7m > \033[0m");
                for (auto& e : m0)
                    printf(" %s:%f,", e.first.c_str(), e.second / 10);
                m0.clear();
            }
        }
    }

    for (auto& tensor : m_data)
    {
        size_t s = 1 + (tensor.m_batch == nullptr ? 0 : tensor.m_batch->size());
        for (size_t i = 0; i < s; i++)
        {
            Tensor& t = !i ? tensor : (*tensor.m_batch)[i - 1];

            switch (t.m_type)
            {
            case TensorDataType::float16:
                m_xnnpack->m_cublas_ops.ensure_is_ready(t.get_vector<uint16_t>().data());
                break;
            case TensorDataType::float32:
                m_xnnpack->m_cublas_ops.ensure_is_ready(t.get_vector<float>().data());
                break;
            }

            if (m_outputs_convert_set.size() && !m_outputs_convert_set.contains(t.m_name))
                continue;

            if (t.m_type == TensorDataType::uint8)
            {
                dequantize(t, TensorDataType::float32);
            }
            else if (t.m_type == TensorDataType::float16)
            {
                auto data = t.get_vector<uint16_t>();
                auto data_fp32 = m_xnnpack->convert<uint16_t, float>(data);
                t.set_vector<float>(std::move(data_fp32));
            }
            else if (t.m_type != TensorDataType::float32 && t.m_type != TensorDataType::int64)
                throw std::invalid_argument("Model::run: invalid type of output tensor.");

            if (t.m_layout == TensorDataLayout::nhwc)
            {
                if (t.m_shape.size() != 4 || t.m_type != TensorDataType::float32)
                    throw std::invalid_argument("Model::run: transpose required but invalid shape and/or data type.");

                auto result = m_xnnpack->transpose(t.m_shape, t.get_vector<float>(), { 0, 3, 1, 2 });

                t.set_vector(std::move(result.second));

                t.m_layout = TensorDataLayout::unspecified;
                t.m_shape = std::move(result.first);
            }
        }
    }

    if (m_first_run && m_use_scaled_dp_attn_op && !m_scaled_dp_attn_op_used)
        throw std::invalid_argument("Model::run: m_use_scaled_dp_attn_op is true but operator not used.");

    m_xnnpack->m_cublas_ops.check_buffers_health();
}

}
