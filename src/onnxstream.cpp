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

#include <xnnpack.h>

static_assert(TENSOR_VECTOR_EXTRA_BYTES == XNN_EXTRA_BYTES);

namespace onnxstream {

class XnnPack
{
public:

    pthreadpool_t threadpool = nullptr;

public:

    XnnPack()
    {
        threadpool = pthreadpool_create(0);
        if (threadpool == nullptr)
            throw std::runtime_error("failed to create threadpool");

        xnn_status status = xnn_initialize(nullptr /* allocator */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to initialize XNNPACK");
    }

    ~XnnPack()
    {
        if (threadpool)
        {
            pthreadpool_destroy(threadpool);
            threadpool = nullptr;
        }
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

    template <typename U, typename T>
    tensor_vector<T> convert(tensor_vector<U>& input)
    {
        const size_t batch_size = input.size();

        tensor_vector<T> output = create_tensor_vector<T>(batch_size);

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
        Qu8MatMulData* qu8_data = nullptr)
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

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_fully_connected_nc_xxx)(size_t, size_t, size_t, size_t, const xnn_ptr_type*, const xnn_ptr_type*, float, float, uint32_t, xnn_code_cache_t, xnn_weights_cache_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_setup_fully_connected_nc_xxx)(xnn_operator_t, size_t, const xnn_ptr_type*, xnn_ptr_type*, pthreadpool_t) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_fully_connected_nc_xxx = &xnn_create_fully_connected_nc_f32;
            xnn_setup_fully_connected_nc_xxx = &xnn_setup_fully_connected_nc_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_fully_connected_nc_xxx = &xnn_create_fully_connected_nc_f16;
            xnn_setup_fully_connected_nc_xxx = &xnn_setup_fully_connected_nc_f16;
        }
        else
        {
            xnn_create_fully_connected_nc_xxx = nullptr;
            xnn_setup_fully_connected_nc_xxx = &xnn_setup_fully_connected_nc_qu8;
        }

        xnn_operator_t fc_op = nullptr;
        xnn_status status;

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

        scope_guard __sg__([&fc_op]() {
            xnn_status status = xnn_delete_operator(fc_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            fc_op = nullptr;
        });

        status = xnn_setup_fully_connected_nc_xxx(
            fc_op /* fully_connected_op */,
            first_shape[0] /* batch_size */,
            first_data /* input */,
            output_data_override ? output_data_override : output_data.data() /* output */,
            threadpool /* threadpool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup fully connected operation");

        status = xnn_run_operator(fc_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run fully connected operator");

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
        enum xnn_status(*xnn_create_sigmoid_nc_xxx)(size_t, size_t, size_t, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_sigmoid_nc_xxx)(xnn_operator_t, size_t, pthreadpool_t) = nullptr;
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
            1 /* channels */, 1 /* input stride */, 1 /* output stride */,
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
    std::pair<std::vector<size_t>, tensor_vector<T>> convolution_nhwc(
        std::vector<size_t>& x_shape, tensor_vector<T>& x_data,
        std::vector<size_t>& w_shape, tensor_vector<T>& w_data,
        std::vector<size_t>& b_shape, U* b_data, size_t b_data_size,
        std::vector<int>& dilations, std::vector<int>& kernel_shape, std::vector<int>& pads, std::vector<int>& strides, int groups,
        Qu8ConvData* qu8_data = nullptr)
    {
        if (x_shape.size() != 4 || w_shape.size() != 4 ||
            dilations.size() != 2 || dilations[0] != dilations[1] ||
            kernel_shape.size() != 2 || pads.size() != 4 ||
            strides.size() != 2 || strides[0] != strides[1])
        {
            throw std::runtime_error("XnnPack::convolution_nhwc_fp32: one or more arguments are invalid.");
        }

        const size_t batch_size = 1;
        const size_t input_height = x_shape[1];
        const size_t input_width = x_shape[2];
        const size_t kernel_height = kernel_shape[0];
        const size_t kernel_width = kernel_shape[1];
        const size_t padding_height = (size_t)(pads[0] + pads[2]);
        const size_t padding_width = (size_t)(pads[1] + pads[3]);
        const size_t subsampling = strides[0];
        const size_t dilation = dilations[0];
        const size_t group_input_channels = x_shape[3];
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

        if (x_data.size() != batch_size * input_height * input_width * input_pixel_stride)
            throw std::runtime_error("XnnPack::convolution_nhwc_fp32: invalid size of X.");
        if (w_data.size() != groups * group_output_channels * kernel_height * kernel_width * group_input_channels)
            throw std::runtime_error("XnnPack::convolution_nhwc_fp32: invalid size of W.");
        if (b_data_size != groups * group_output_channels)
            throw std::runtime_error("XnnPack::convolution_nhwc_fp32: invalid size of B.");

        std::vector<size_t> output_shape({ 1, output_height, output_width, group_output_channels });

        tensor_vector<T> output_data = create_tensor_vector<T>(output_elements);

        typedef
            typename std::conditional<std::is_same<T, float>::value, float,
            typename std::conditional<std::is_same<T, uint16_t>::value, void,
            uint8_t>::type>::type xnn_ptr_type;

        enum xnn_status(*xnn_create_convolution2d_nhwc_xxx)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t, size_t, const xnn_ptr_type*, const xnn_ptr_type*, float, float, uint32_t, xnn_code_cache_t, xnn_weights_cache_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_reshape_convolution2d_nhwc_xxx)(xnn_operator_t, size_t, size_t, size_t, size_t*, size_t*, pthreadpool_t) = nullptr;
        enum xnn_status(*xnn_setup_convolution2d_nhwc_xxx)(xnn_operator_t, const xnn_ptr_type*, xnn_ptr_type*) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_convolution2d_nhwc_xxx = &xnn_create_convolution2d_nhwc_f32;
            xnn_reshape_convolution2d_nhwc_xxx = &xnn_reshape_convolution2d_nhwc_f32;
            xnn_setup_convolution2d_nhwc_xxx = &xnn_setup_convolution2d_nhwc_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_convolution2d_nhwc_xxx = &xnn_create_convolution2d_nhwc_f16;
            xnn_reshape_convolution2d_nhwc_xxx = &xnn_reshape_convolution2d_nhwc_f16;
            xnn_setup_convolution2d_nhwc_xxx = &xnn_setup_convolution2d_nhwc_f16;
        }
        else
        {
            xnn_create_convolution2d_nhwc_xxx = nullptr;
            xnn_reshape_convolution2d_nhwc_xxx = &xnn_reshape_convolution2d_nhwc_qu8;
            xnn_setup_convolution2d_nhwc_xxx = &xnn_setup_convolution2d_nhwc_qu8;
        }

        xnn_operator_t convolution_op = nullptr;
        xnn_status status;

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

        scope_guard __sg__([&convolution_op]() {
            xnn_status status = xnn_delete_operator(convolution_op);
            if (status != xnn_status_success)
                throw std::runtime_error("failed to delete operator");
            convolution_op = nullptr;
        });

        status = xnn_reshape_convolution2d_nhwc_xxx(
            convolution_op, batch_size, input_height, input_width,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            /*threadpool=*/ threadpool);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to reshape Convolution operator");

        status = xnn_setup_convolution2d_nhwc_xxx(
            convolution_op,
            x_data.data(), output_data.data());
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup FP32 Convolution operator");

        status = xnn_run_operator(convolution_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run FP32 Convolution operator");

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
        enum xnn_status(*xnn_setup_transpose_nd_xxx)(xnn_operator_t, const void*, void*, size_t, const size_t*, const size_t*, pthreadpool_t) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_transpose_nd_xxx = &xnn_create_transpose_nd_x32;
            xnn_setup_transpose_nd_xxx = &xnn_setup_transpose_nd_x32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_transpose_nd_xxx = &xnn_create_transpose_nd_x16;
            xnn_setup_transpose_nd_xxx = &xnn_setup_transpose_nd_x16;
        }
        else if constexpr (std::is_same<T, uint8_t>::value)
        {
            xnn_create_transpose_nd_xxx = &xnn_create_transpose_nd_x8;
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

        status = xnn_setup_transpose_nd_xxx(
            transpose_op,
            input_data.data() /* input */,
            output_data.data() /* output */,
            perm.size() /* num_dims */,
            input_shape.data() /* shape */,
            perm.data() /* perm */,
            threadpool /* threadpool */);
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
        Qu8SoftmaxData* qu8_data = nullptr)
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

        enum xnn_status(*xnn_create_softmax_nc_xxx)(size_t, size_t, size_t, uint32_t, xnn_operator_t*) = nullptr;
        enum xnn_status(*xnn_setup_softmax_nc_xxx)(xnn_operator_t, size_t, const xnn_ptr_type*, xnn_ptr_type*, pthreadpool_t) = nullptr;

        if constexpr (std::is_same<T, float>::value)
        {
            xnn_create_softmax_nc_xxx = &xnn_create_softmax_nc_f32;
            xnn_setup_softmax_nc_xxx = &xnn_setup_softmax_nc_f32;
        }
        else if constexpr (std::is_same<T, uint16_t>::value)
        {
            xnn_create_softmax_nc_xxx = &xnn_create_softmax_nc_f16;
            xnn_setup_softmax_nc_xxx = &xnn_setup_softmax_nc_f16;
        }
        else
        {
            xnn_create_softmax_nc_xxx = nullptr;
            xnn_setup_softmax_nc_xxx = &xnn_setup_softmax_nc_qu8;
        }

        size_t channels = input_shape.back();
        size_t batch_size = output_size / channels;

        xnn_operator_t softmax_op = nullptr;
        xnn_status status;

        if constexpr (!std::is_same<T, uint8_t>::value)
        {
            status = xnn_create_softmax_nc_xxx(
                channels /* channels */, channels /* input stride */, channels /* output stride */,
                0 /* flags */, &softmax_op);
        }
        else
        {
            status = xnn_create_softmax_nc_qu8(
                channels /* channels */, channels /* input stride */, channels /* output stride */,
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

        status = xnn_setup_softmax_nc_xxx(
            softmax_op, batch_size /* batch_size */,
            input_data, output_data_override ? output_data_override : output_data.data(),
            threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to setup softmax operator");

        status = xnn_run_operator(softmax_op, threadpool /* thread pool */);
        if (status != xnn_status_success)
            throw std::runtime_error("failed to run softmax operator");

        return std::pair<std::vector<size_t>, tensor_vector<T>>(
            std::move(output_shape), std::move(output_data));
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

Model::Model()
{
    m_xnnpack = new XnnPack();
}

Model::~Model()
{
    if (m_xnnpack)
        delete m_xnnpack;
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

std::string Model::next_line()
{
    return next_file_line(m_model, m_pos);
}

std::optional<Operation> Model::next_op()
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
            if (i <= 0)
                throw std::invalid_argument("Model::parse_tensor_string: invalid shape.");

            t.m_shape.push_back(i);
        }
    }

    return t;
}

Tensor& Model::get_tensor_data(Tensor& t, bool make_copy /*= false*/, bool requires_float /*= false*/, TensorDataLayout required_layout /*= TensorDataLayout::unspecified*/, bool accepts_multipart /*= false*/)
{
    if (m_pow_requires_float && m_ops_queue.size() && m_ops_queue[0].m_type == "Pow")
        requires_float = true;

    if (t.m_type != TensorDataType::none)
    {
        std::string fn = t.m_name;

        if (required_layout == TensorDataLayout::nhwc)
        {
            auto lpos = fn.find("_nchw.bin");
            if (lpos == std::string::npos)
            {
                throw std::invalid_argument("Model::get_tensor_data: unable to determine tensor data file compatible with required_layout.");
            }
            else
            {
                if (t.m_layout != TensorDataLayout::unspecified)
                    throw std::invalid_argument("Model::get_tensor_data: tensor data layout already set.");
                else
                    t.m_layout = TensorDataLayout::nhwc;

                if (t.m_shape.size() != 4)
                    throw std::invalid_argument("Model::get_tensor_data: layout is nhwc but invalid shape.");
                else
                    t.m_shape = { t.m_shape[0], t.m_shape[2], t.m_shape[3], t.m_shape[1] };

                fn = fn.substr(0, lpos) + "_nhwc.bin";
            }
        }

        switch (t.m_type)
        {
        case TensorDataType::uint8:
        {
            auto data = get_wp()->get_uint8(fn);
            t.set_vector(std::move(data));
            break;
        }
        case TensorDataType::float16:
        {
            auto data = get_wp()->get_float16(fn);
            t.set_vector(std::move(data));
            break;
        }
        case TensorDataType::float32:
        {
            auto data = get_wp()->get_float32(fn);
            t.set_vector(std::move(data));
            break;
        }
        case TensorDataType::int64:
        {
            auto data = get_wp()->get_int64(fn);
            t.set_vector(std::move(data));
            break;
        }
        default:
            throw std::invalid_argument("Model::get_tensor_data: unsupported tensor data format.");
        }
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
        }
        else
        {
            if (tensor_ptr->m_next_part)
                throw std::runtime_error("Model::get_tensor_data: multipart tensors cannot have multiple references (not implemented).");

            if (!make_copy)
            {
                t.m_data = tensor_ptr->m_data;
                t.m_type = tensor_ptr->m_type;
                t.m_layout = tensor_ptr->m_layout;
                t.m_shape = tensor_ptr->m_shape;
                t.m_scale = tensor_ptr->m_scale;
                t.m_zero_point = tensor_ptr->m_zero_point;
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
            }
        }
    }

    if (t.m_next_part && !accepts_multipart)
        throw std::invalid_argument("Model::get_tensor_data: multipart tensor but accepts_multipart == false (" + (!m_ops_queue.size() ? std::string("?") : m_ops_queue[0].m_type) + ").");

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

    if (from_shape <= 0 || from_shape != size)
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

    std::vector<size_t> transpose_perm;

    if (required_layout == TensorDataLayout::nhwc && t.m_layout == TensorDataLayout::unspecified)
    {
        transpose_perm = { 0, 2, 3, 1 };
    }
    else if (required_layout == TensorDataLayout::unspecified && t.m_layout == TensorDataLayout::nhwc)
    {
        transpose_perm = { 0, 3, 1, 2 };
    }

    if (transpose_perm.size())
    {
        if (t.m_shape.size() != 4)
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

    return t;
}

void Model::push_tensor(Tensor&& t, bool force_quantization /*= false*/)
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

    if (m_ops_queue.size() >= 2 && m_ops_queue[0].m_output.size() == 1)
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

        for (Tensor& first : m_data)
            if (first.m_name == t.m_name)
            {
                Tensor* last = &first;
                while (last->m_next_part)
                    last = last->m_next_part.get();

                last->m_next_part = std::make_unique<Tensor>(std::move(t));

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

Tensor& Model::get_multipart_input(Tensor& t, size_t i, TensorDataType base_type)
{
    if (!t.m_next_part)
    {
        return t;
    }

    int count = i;
    Tensor* curr = &t;
    Tensor* prev = curr;
    while (count > 0 && curr->m_next_part)
    {
        if (prev->m_type != TensorDataType::none) prev->reset_data();
        prev = curr;
        curr = curr->m_next_part.get();
        count--;
    }
    if (count)
        throw std::invalid_argument("Model::get_multipart_input: wrong number of tensors in multipart tensor.");

    if (curr->m_type == TensorDataType::uint8)
        dequantize(*curr, base_type);

    return *curr;
}

void Model::push_multipart_tensor(Tensor& output, bool is_multipart)
{
    Tensor copy = output.get_copy_without_data();
    push_tensor(std::move(output), is_multipart && m_do_multipart_quantization /* force_quantization */);
    output = std::move(copy);
}

size_t Model::get_multipart_dimension(Tensor& t)
{
    size_t n = 1;

    Tensor* ptr = t.m_next_part.get();
    while (ptr)
    {
        ptr = ptr->m_next_part.get();
        n++;
    }

    return n;
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

void Model::run()
{
    if (m_intermediate_refs_copy.size() == 0)
    {
        m_pos = 0;

        while (auto op = next_op())
        {
            for (auto& input_tensor : op->m_input)
            {
                if (input_tensor.m_name.size() != 0)
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

        for (auto& name : m_extra_outputs)
            m_intermediate_refs[name]++;

        m_intermediate_refs_copy = m_intermediate_refs;
    }
    else
    {
        m_intermediate_refs = m_intermediate_refs_copy;

        m_ops_printf_index = 0;

        get_wp()->on_restart();
    }

    m_pos = 0;

    while (true)
    {
        if (m_ops_queue.size())
            m_ops_queue.erase(m_ops_queue.begin());

        const size_t ops_to_read = 5;

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

        if (m_fuse_ops_in_attention)
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
                        if (m_force_uint8_storage_set.contains(t.m_name))
                        {
                            quantize(t, 0.001, 0.001);
                        }
                        else
                        {
                            auto data = t.get_vector<float>();
                            auto data_fp16 = m_xnnpack->convert<float, uint16_t>(data);
                            t.set_vector<uint16_t>(std::move(data_fp16));
                        }
                    }
                }
            }
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

            if (!compare_shapes(data.m_shape, output.m_shape))
                throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

            output.m_type = data.m_type;
            output.m_data = std::move(data.m_data);

            push_tensor(std::move(output));
        }
        else if (op.m_type == "Mul")
        {
            if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
            if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

            auto& input_0_base = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::unspecified /* required_layout */, true /* accepts_multipart */);
            auto& input_1 = get_tensor_data(op.m_input[1]);
            auto& output = op.m_output[0];

            if (input_0_base.m_next_part)
                output.m_shape[0] = 1;

            TensorDataType base_type = input_0_base.m_type;

            size_t n = get_multipart_dimension(input_0_base);

            for (size_t i = 0; i < n; i++)
            {
                Tensor& input_0 = get_multipart_input(input_0_base, i, base_type);

                if (input_0.m_type == TensorDataType::float32)
                {
                    if (input_0.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<float>();
                    auto& input_1_data = input_1.get_vector<float>();

                    auto result = m_xnnpack->multiply(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data());

                    if (!compare_shapes(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input_0.m_type == TensorDataType::float16)
                {
                    if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                    if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                    auto& input_0_data = input_0.get_vector<uint16_t>();
                    auto& input_1_data = input_1.get_vector<uint16_t>();

                    auto result = m_xnnpack->multiply(input_0.m_shape, input_0_data.data(), input_1.m_shape, input_1_data.data());

                    if (!compare_shapes(result.first, output.m_shape))
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

                    if (!compare_shapes(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));

                    output.m_scale = s.first;
                    output.m_zero_point = s.second;
                }

                push_multipart_tensor(output, n > 1 /* is_multipart */);
            }
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

            if (!compare_shapes(input.m_shape, output.m_shape))
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
                if (!compare_shapes(t.m_shape, op.m_input[0].m_shape, axis))
                    throw std::invalid_argument(op.m_type + ": invalid shape of one or more inputs.");

                size_t num = 1;
                for (int i = axis; i < t.m_shape.size(); i++)
                    num *= t.m_shape[i];

                final_dim += t.m_shape[axis];
                output_stride += num;

                auto& input = get_tensor_data(t);
                if (!sizeof_element)
                {
                    if (input.m_type == TensorDataType::float16)
                        sizeof_element = sizeof(uint16_t);
                    else if (input.m_type == TensorDataType::float32)
                        sizeof_element = sizeof(float);
                    else
                        throw std::invalid_argument(op.m_type + ": wrong data type of input.");
                }
                else
                {
                    if (
                        (input.m_type != TensorDataType::float16 && input.m_type != TensorDataType::float32) ||
                        (input.m_type == TensorDataType::float16 && sizeof_element != sizeof(uint16_t)) ||
                        (input.m_type == TensorDataType::float32 && sizeof_element != sizeof(float))
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
                else
                {
                    auto& input_data = input.get_vector<uint16_t>();
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
            else
            {
                tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);
                output.set_vector(std::move(output_data));
                ptr_output = output.get_vector<uint16_t>().data();
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

            if (!compare_shapes(output_shape, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }
            else if (input.m_type == TensorDataType::float16)
            {
                if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                auto& input_data = input.get_vector<uint16_t>();

                auto result = m_xnnpack->sigmoid(input.m_shape, input_data);

                if (!compare_shapes(result.first, output.m_shape))
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

                if (!compare_shapes(input.m_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
            }

            push_tensor(std::move(output));
        }
        else if (op.m_type == "Conv")
        {
            if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs. 2 inputs case not implemented.");
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

            if (!are_all_equal(dilations, { 1, 1 }))
                throw std::invalid_argument(op.m_type + ": invalid dilations attribute value (not implemented).");
            /*if (!are_all_equal(pads, {1, 1, 1, 1}))
                throw std::invalid_argument(op.m_type + ": invalid pads attribute value (not implemented).");*/
                /*if (!are_all_equal(strides, {1, 1}))
                    throw std::invalid_argument(op.m_type + ": invalid strides attribute value (not implemented).");*/
            if (group != 1)
                throw std::invalid_argument(op.m_type + ": invalid group attribute value (not implemented).");

            auto& x = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::nhwc /* required_layout */);
            auto& w = get_tensor_data(op.m_input[1], false /* make_copy */, false /* requires_float */, TensorDataLayout::nhwc /* required_layout */);
            auto& b = get_tensor_data(op.m_input[2], true /* make_copy */);
            auto& output = op.m_output[0];

            if (x.m_layout != TensorDataLayout::nhwc) throw std::invalid_argument(op.m_type + ": wrong layout of X.");
            if (w.m_layout != TensorDataLayout::nhwc) throw std::invalid_argument(op.m_type + ": wrong layout of W.");

            if (w.m_shape.size() != 4 || !are_all_equal(kernel_shape, { w.m_shape[1], w.m_shape[2] }))
                throw std::invalid_argument(op.m_type + ": invalid shape of W or invalid kernel_shape (not implemented?).");

            std::vector<size_t> result_first;

            if (x.m_type == TensorDataType::float32)
            {
                if (x.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of X.");
                if (w.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of W.");
                if (b.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of B.");

                auto& x_data = x.get_vector<float>();
                auto& w_data = w.get_vector<float>();
                auto& b_data = b.get_vector<float>();

                auto result = m_xnnpack->convolution_nhwc<float, float>(
                    x.m_shape, x_data,
                    w.m_shape, w_data,
                    b.m_shape, b_data.data(), b_data.size(),
                    dilations, kernel_shape, pads, strides, group);

                result_first = std::move(result.first);

                output.set_vector(std::move(result.second));
            }
            else if (x.m_type == TensorDataType::float16)
            {
                if (x.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of X.");
                if (w.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of W.");
                if (b.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of B.");

                auto& x_data = x.get_vector<uint16_t>();
                auto& w_data = w.get_vector<uint16_t>();
                auto& b_data = b.get_vector<uint16_t>();

                auto result = m_xnnpack->convolution_nhwc<uint16_t, uint16_t>(
                    x.m_shape, x_data,
                    w.m_shape, w_data,
                    b.m_shape, b_data.data(), b_data.size(),
                    dilations, kernel_shape, pads, strides, group);

                result_first = std::move(result.first);

                output.set_vector(std::move(result.second));
            }
            else
            {
                if (x.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of X.");
                if (w.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of W.");
                if (b.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of B.");

                auto& x_data = x.get_vector<uint8_t>();
                auto& w_data = w.get_vector<uint8_t>();
                auto& b_data = b.get_vector<float>();

                struct context
                {
                    float* ptr;
                    float scale;
                };

                context prl_ctx;

                prl_ctx.ptr = b_data.data();
                prl_ctx.scale = x.m_scale * w.m_scale;

                void (*pfn)(context*, size_t) = nullptr;

                pfn = [](context* _, size_t i)
                {
                    *(int32_t*)(_->ptr + i) = (int32_t)(_->ptr[i] / _->scale);
                };

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, b_data.size());

                XnnPack::Qu8ConvData qu8_data;

                if (!m_range_data.count(op.m_name)) throw std::invalid_argument(op.m_type + ": range data not found.");

                auto s = range_to_scale(m_range_data[op.m_name]);

                qu8_data.input_zero_point = x.m_zero_point;
                qu8_data.input_scale = x.m_scale;
                qu8_data.kernel_zero_point = w.m_zero_point;
                qu8_data.kernel_scale = w.m_scale;
                qu8_data.output_zero_point = s.second;
                qu8_data.output_scale = s.first;

                auto result = m_xnnpack->convolution_nhwc<uint8_t, int32_t>(
                    x.m_shape, x_data,
                    w.m_shape, w_data,
                    b.m_shape, (int32_t*)b_data.data(), b_data.size(),
                    dilations, kernel_shape, pads, strides, group,
                    &qu8_data);

                result_first = std::move(result.first);

                output.set_vector(std::move(result.second));

                output.m_scale = s.first;
                output.m_zero_point = s.second;
            }

            if (result_first.size() != 4 ||
                !compare_shapes(output.m_shape, { result_first[0], result_first[3], result_first[1], result_first[2] }))
            {
                throw std::invalid_argument(op.m_type + ": unexpected shape of output.");
            }

            output.m_layout = TensorDataLayout::nhwc;
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

            if (!compare_shapes(output_shape, output.m_shape))
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

            if (!compare_shapes(input.m_shape, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }
            else
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

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));

                output.m_scale = s.first;
                output.m_zero_point = s.second;
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

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }
            else if (input.m_type == TensorDataType::float16)
            {
                if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                auto& input_data = input.get_vector<uint16_t>();

                auto result = m_xnnpack->transpose(input.m_shape, input_data, perm);

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }
            else
            {
                if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                auto& input_data = input.get_vector<uint8_t>();

                auto result = m_xnnpack->transpose(input.m_shape, input_data, perm);

                if (!compare_shapes(result.first, output.m_shape))
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

            for (auto& a : op.m_attributes)
                if (a.first == "axes")
                    axes = string_to_int_vec<int>(a.second);
                else
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

            if (axes.size() != 1)
                throw std::invalid_argument(op.m_type + ": reduce supported on 1 axis only (not implemented).");

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

            if (!compare_shapes(output_shape, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }
            else
            {
                if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                auto& input_0_data = input_0.get_vector<uint16_t>();
                auto& input_1_data = input_1.get_vector<uint16_t>();

                auto result = m_xnnpack->subtract(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
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

            if (!compare_shapes(input.m_shape, output.m_shape))
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

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }
            else
            {
                if (input_0.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");

                auto& input_0_data = input_0.get_vector<uint16_t>();
                auto& input_1_data = input_1.get_vector<uint16_t>();

                auto result = m_xnnpack->divide(input_0.m_shape, input_0_data, input_1.m_shape, input_1_data);

                if (!compare_shapes(result.first, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                output.set_vector(std::move(result.second));
            }

            push_tensor(std::move(output));
        }
        else if (op.m_type == "MatMul")
        {
            if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
            if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

            auto& input_0 = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::unspecified /* required_layout */, true /* accepts_multipart */);
            auto& input_1 = get_tensor_data(op.m_input[1]);
            auto& output = op.m_output[0];

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
                first_shape_is_2d = true;
            }

            if (input_0.m_shape.size() != 3)
                throw std::invalid_argument(op.m_type + ": shape of input 0 must have 3 dimensions (not implemented).");

            size_t n = input_0.m_shape[0];

            bool is_multipart_input = input_0.m_next_part.get() != nullptr;

            if (is_multipart_input)
            {
                if (n != 1) throw std::invalid_argument(op.m_type + ": invalid shape of multipart input.");
                n = get_multipart_dimension(input_0);
            }

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

            if (!compare_shapes(output_shape, output.m_shape))
                throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

            size_t output_num_els = 1;
            for (auto& s : output_shape)
                output_num_els *= s;

            bool is_multipart_output = false;

            if (m_multipart_threshold != -1 && output_num_els >= m_multipart_threshold && n > 1)
            {
                output_shape[0] = 1;
                output.m_shape[0] = 1;
                output_num_els /= n;
                is_multipart_output = true;
            }

            void* input_0_ptr = nullptr;
            void* input_1_ptr = nullptr;
            void* output_ptr = nullptr;
            size_t sizeof_element = 0;

            TensorDataType base_type = input_0.m_type;

            for (size_t i = 0; i < n; i++)
            {
                Tensor& input_0_curr = get_multipart_input(input_0, i, base_type);

                if (input_0_curr.m_type == TensorDataType::float32)
                {
                    sizeof_element = sizeof(float);

                    if (i == 0)
                    {
                        if (input_1.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                        auto& input_1_data = input_1.get_vector<float>();
                        input_1_ptr = input_1_data.data();
                    }

                    if (is_multipart_input || i == 0)
                    {
                        if (input_0_curr.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                        auto& input_0_data = input_0_curr.get_vector<float>();
                        input_0_ptr = input_0_data.data();
                    }

                    if (is_multipart_output || i == 0)
                    {
                        tensor_vector<float> temp = create_tensor_vector<float>(output_num_els);
                        output.set_vector(std::move(temp));
                        output_ptr = output.get_vector<float>().data();
                    }
                }
                else if (input_0_curr.m_type == TensorDataType::float16)
                {
                    sizeof_element = sizeof(uint16_t);

                    if (i == 0)
                    {
                        if (input_1.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                        auto& input_1_data = input_1.get_vector<uint16_t>();
                        input_1_ptr = input_1_data.data();
                    }

                    if (is_multipart_input || i == 0)
                    {
                        if (input_0_curr.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                        auto& input_0_data = input_0_curr.get_vector<uint16_t>();
                        input_0_ptr = input_0_data.data();
                    }

                    if (is_multipart_output || i == 0)
                    {
                        tensor_vector<uint16_t> temp = create_tensor_vector<uint16_t>(output_num_els);
                        output.set_vector(std::move(temp));
                        output_ptr = output.get_vector<uint16_t>().data();
                    }
                }
                else
                {
                    sizeof_element = sizeof(uint8_t);

                    if (i == 0)
                    {
                        if (input_1.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 1.");
                        auto& input_1_data = input_1.get_vector<uint8_t>();
                        input_1_ptr = input_1_data.data();
                    }

                    if (is_multipart_input || i == 0)
                    {
                        if (input_0_curr.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input 0.");
                        auto& input_0_data = input_0_curr.get_vector<uint8_t>();
                        input_0_ptr = input_0_data.data();
                    }

                    if (is_multipart_output || i == 0)
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

                std::vector<size_t> result_first;

                if (sizeof_element == sizeof(float))
                {
                    auto result = m_xnnpack->matrix_multiply<float, float>(
                        { input_0_curr.m_shape[1], input_0_curr.m_shape[2] },
                        (float*)input_0_ptr,
                        { input_1.m_shape[1], input_1.m_shape[2] },
                        (float*)input_1_ptr,
                        nullptr, nullptr,
                        (float*)output_ptr);

                    result_first = std::move(result.first);
                }
                else if (sizeof_element == sizeof(uint16_t))
                {
                    auto result = m_xnnpack->matrix_multiply<uint16_t, uint16_t>(
                        { input_0_curr.m_shape[1], input_0_curr.m_shape[2] },
                        (uint16_t*)input_0_ptr,
                        { input_1.m_shape[1], input_1.m_shape[2] },
                        (uint16_t*)input_1_ptr,
                        nullptr, nullptr,
                        (uint16_t*)output_ptr);

                    result_first = std::move(result.first);
                }
                else
                {
                    XnnPack::Qu8MatMulData qu8_data;

                    qu8_data.input_zero_point = input_0_curr.m_zero_point;
                    qu8_data.input_scale = input_0_curr.m_scale;
                    qu8_data.kernel_zero_point = input_1.m_zero_point;
                    qu8_data.kernel_scale = input_1.m_scale;
                    qu8_data.output_zero_point = output.m_zero_point;
                    qu8_data.output_scale = output.m_scale;

                    auto result = m_xnnpack->matrix_multiply<uint8_t, int32_t>(
                        { input_0_curr.m_shape[1], input_0_curr.m_shape[2] },
                        (uint8_t*)input_0_ptr,
                        { input_1.m_shape[1], input_1.m_shape[2] },
                        (uint8_t*)input_1_ptr,
                        nullptr, nullptr,
                        (uint8_t*)output_ptr,
                        &qu8_data);

                    result_first = std::move(result.first);
                }

                if (result_first.size() != 2)
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output of matrix_multiply_fp32.");

                input_0_ptr = (uint8_t*)input_0_ptr + input_0_curr.m_shape[1] * input_0_curr.m_shape[2] * sizeof_element;
                input_1_ptr = (uint8_t*)input_1_ptr + input_1.m_shape[1] * input_1.m_shape[2] * sizeof_element;
                output_ptr = (uint8_t*)output_ptr + result_first[0] * result_first[1] * sizeof_element;

                if (!is_multipart_output)
                {
                    if (i == n - 1)
                        push_tensor(std::move(output));
                }
                else
                {
                    push_multipart_tensor(output, true /* is_multipart */);
                }
            }
        }
        else if (op.m_type == "Softmax")
        {
            if (op.m_input.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
            if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

            auto& input_base = get_tensor_data(op.m_input[0], false /* make_copy */, false /* requires_float */, TensorDataLayout::unspecified /* required_layout */, true /* accepts_multipart */);
            auto& output = op.m_output[0];

            if (input_base.m_next_part)
                output.m_shape[0] = 1;

            int axis = -1;

            for (auto& a : op.m_attributes)
                if (a.first == "axis")
                    axis = std::stoi(a.second);
                else
                    throw std::invalid_argument(op.m_type + ": unrecognized attribute: " + a.first + ".");

            if (axis < 0)
                axis = (int)input_base.m_shape.size() + axis;
            if (axis < 0 || axis >= input_base.m_shape.size())
                throw std::invalid_argument(op.m_type + ": invalid axis attribute.");

            if (axis != input_base.m_shape.size() - 1)
                throw std::invalid_argument(op.m_type + ": softmax supported on last axis only (not implemented).");

            TensorDataType base_type = input_base.m_type;

            size_t n = get_multipart_dimension(input_base);

            for (size_t i = 0; i < n; i++)
            {
                Tensor& input = get_multipart_input(input_base, i, base_type);

                if (input.m_type == TensorDataType::float32)
                {
                    if (input.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<float>();

                    auto result = m_xnnpack->softmax(input.m_shape, input_data.data());

                    if (!compare_shapes(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else if (input.m_type == TensorDataType::float16)
                {
                    if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint16_t>();

                    auto result = m_xnnpack->softmax(input.m_shape, input_data.data());

                    if (!compare_shapes(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }
                else
                {
                    if (input.m_type != TensorDataType::uint8) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                    auto& input_data = input.get_vector<uint8_t>();

                    XnnPack::Qu8SoftmaxData qu8_data;

                    qu8_data.input_scale = input.m_scale;
                    qu8_data.output_zero_point = output.m_zero_point = 0;
                    qu8_data.output_scale = output.m_scale = 0x1.0p-8f;

                    auto result = m_xnnpack->softmax<uint8_t>(input.m_shape, input_data.data(), nullptr, &qu8_data);

                    if (!compare_shapes(result.first, output.m_shape))
                        throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                    output.set_vector(std::move(result.second));
                }

                push_multipart_tensor(output, n > 1 /* is_multipart */);
            }
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

            if (axis != input.m_shape.size() - 1)
                throw std::invalid_argument(op.m_type + ": split supported on last axis only (not implemented).");

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

                size_t ops_num = output_num_els / dim;

                struct context
                {
                    size_t input_index, input_stride, sizeof_element;
                    int64_t dim;
                    void* input_ptr, * output_ptr;
                };

                context prl_ctx;

                prl_ctx.input_index = start_input_index;
                prl_ctx.input_stride = input.m_shape[axis];
                prl_ctx.sizeof_element = sizeof_element;
                prl_ctx.dim = dim;
                prl_ctx.input_ptr = input_ptr;
                prl_ctx.output_ptr = output_ptr;

                void (*pfn)(context*, size_t) = [](context* _, size_t j)
                {
                    memcpy(
                        (uint8_t*)_->output_ptr + j * _->dim * _->sizeof_element,
                        (uint8_t*)_->input_ptr + (_->input_index + j * _->input_stride) * _->sizeof_element,
                        _->dim * _->sizeof_element);
                };

                m_xnnpack->parallelize((void*)pfn, &prl_ctx, ops_num);

                if (!compare_shapes(output_shape, output.m_shape))
                    throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

                push_tensor(std::move(output));

                start_input_index += dim;
            }
        }
        else if (op.m_type == "Resize")
        {
            if (op.m_input.size() != 3) throw std::invalid_argument(op.m_type + ": wrong number of inputs (not implemented).");
            if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

            if (op.m_input[1].m_name.size() != 0) throw std::invalid_argument(op.m_type + ": 'roi' input not supported (not implemented).");

            auto& input = get_tensor_data(op.m_input[0]);
            auto& scales = get_tensor_data(op.m_input[2], false /* make_copy */, true /* requires_float */);
            auto& output = op.m_output[0];

            if (scales.m_type != TensorDataType::float32) throw std::invalid_argument(op.m_type + ": wrong data type of scales.");

            auto& scales_data = scales.get_vector<float>();

            if (input.m_shape.size() != 4) throw std::invalid_argument(op.m_type + ": input must be 4D (not implemented).");
            if (scales_data.size() != input.m_shape.size()) throw std::invalid_argument(op.m_type + ": invalid data size of scales.");
            if (input.m_shape[0] != 1) throw std::invalid_argument(op.m_type + ": first dimension of input's shape must be 1 (not implemented).");
            if (scales_data[0] != 1 || scales_data[1] != 1) throw std::invalid_argument(op.m_type + ": first and second value of scales must be 1 (not implemented).");

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

            std::vector<size_t> output_shape;
            for (size_t i = 0; i < input.m_shape.size(); i++)
                output_shape.push_back(input.m_shape[i] * scales_data[i]);

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

                        size_t final_output = pos_output + y_output * _->x_output_size + x_output;
                        size_t final_input = pos_input + y_input * _->x_input_size + x_input;

                        copy((uint8_t*)_->output_data_ptr + final_output * _->sizeof_element,
                            (uint8_t*)_->input_data_ptr + final_input * _->sizeof_element);
                    }
            };

            m_xnnpack->parallelize((void*)pfn, &prl_ctx, prl_ctx.n_size);

            if (!compare_shapes(output_shape, output.m_shape))
                throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

            push_tensor(std::move(output));
        }
        else if (op.m_type == "Gather")
        {
            if (op.m_input.size() != 2) throw std::invalid_argument(op.m_type + ": wrong number of inputs.");
            if (op.m_output.size() != 1) throw std::invalid_argument(op.m_type + ": wrong number of outputs.");

            if (op.m_attributes.size())
                throw std::invalid_argument(op.m_type + ": unrecognized attribute (not implemented).");

            auto& input = get_tensor_data(op.m_input[0]);
            auto& indices = get_tensor_data(op.m_input[1]);
            auto& output = op.m_output[0];

            if (indices.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of indices.");

            bool indices_shape_1d = false;

            if (indices.m_shape.size() == 1)
            {
                indices.m_shape.insert(indices.m_shape.begin(), 1);
                indices_shape_1d = true;
            }

            if (indices.m_shape.size() != 2 || indices.m_shape[0] != 1)
                throw std::invalid_argument(op.m_type + ": shape of indices must be (1,D) (not implemented).");
            if (input.m_shape.size() != 2)
                throw std::invalid_argument(op.m_type + ": input must be 2D (not implemented).");

            auto& indices_data = indices.get_vector<int64_t>();

            std::vector<size_t> output_shape({ 1, indices.m_shape[1], input.m_shape[1] });

            if (indices_shape_1d)
                output_shape.erase(output_shape.begin());

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
            else
            {
                if (input.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of input.");

                auto& input_data = input.get_vector<uint16_t>();

                tensor_vector<uint16_t> output_data = create_tensor_vector<uint16_t>(output_num_els);
                output.set_vector(std::move(output_data));

                sizeof_element = sizeof(uint16_t);
                ptr_output = output.get_vector<uint16_t>().data();
                ptr_input = input_data.data();
            }

            struct context
            {
                std::vector<size_t>* input_shape;
                tensor_vector<int64_t>* indices_data;
                bool error_0;
                size_t sizeof_element;
                void* ptr_input, * ptr_output;
            };

            context prl_ctx;

            prl_ctx.input_shape = &input.m_shape;
            prl_ctx.indices_data = &indices_data;
            prl_ctx.error_0 = false;
            prl_ctx.sizeof_element = sizeof_element;
            prl_ctx.ptr_input = ptr_input;
            prl_ctx.ptr_output = ptr_output;

            void (*pfn)(context*, size_t) = [](context* _, size_t i)
            {
                int64_t index = (*_->indices_data)[i];

                if (index < 0)
                    index = (int64_t)(*_->input_shape)[0] + index;
                if (index < 0 || index >= (int64_t)(*_->input_shape)[0])
                {
                    _->error_0 = true;
                    return;
                }

                const size_t els = (*_->input_shape)[1];

                memcpy((uint8_t*)_->ptr_output + i * els * _->sizeof_element,
                    (uint8_t*)_->ptr_input + index * els * _->sizeof_element,
                    els * _->sizeof_element);
            };

            m_xnnpack->parallelize((void*)pfn, &prl_ctx, indices_data.size());

            if (prl_ctx.error_0)
                throw std::invalid_argument(op.m_type + ": invalid index in indices.");

            if (!compare_shapes(output_shape, output.m_shape))
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
            auto& output = op.m_output[0];

            if (starts.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of starts.");
            if (ends.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of ends.");

            if (!are_all_equal(starts.m_shape, { 1 }))
                throw std::invalid_argument(op.m_type + ": unsupported shape of starts (not implemented).");
            if (!are_all_equal(ends.m_shape, { 1 }))
                throw std::invalid_argument(op.m_type + ": unsupported shape of ends (not implemented).");

            auto& starts_data = starts.get_vector<int64_t>();
            auto& ends_data = ends.get_vector<int64_t>();

            if (op.m_input.size() > 3)
            {
                auto& axes = get_tensor_data(op.m_input[3]);

                if (axes.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of axes.");

                if (!are_all_equal(axes.m_shape, { 1 }))
                    throw std::invalid_argument(op.m_type + ": unsupported shape of axes (not implemented).");

                auto& axes_data = axes.get_vector<int64_t>();

                int axis = (int)axes_data[0];
                int rank = data.m_shape.size();

                if (axis < 0)
                    axis = rank + axis;
                if (axis < 0 || axis >= rank)
                    throw std::invalid_argument(op.m_type + ": invalid axis in axes.");

                if (axis != rank - 1)
                    throw std::invalid_argument(op.m_type + ": unsupported axes value(s): slice supported on last axis only (not implemented).");
            }

            if (op.m_input.size() > 4)
            {
                auto& steps = get_tensor_data(op.m_input[4]);

                if (steps.m_type != TensorDataType::int64) throw std::invalid_argument(op.m_type + ": wrong data type of steps.");

                if (!are_all_equal(steps.m_shape, { 1 }))
                    throw std::invalid_argument(op.m_type + ": unsupported shape of steps (not implemented).");

                auto& steps_data = steps.get_vector<int64_t>();

                if (steps_data[0] != 1)
                    throw std::invalid_argument(op.m_type + ": unsupported steps value(s) (not implemented).");
            }

            int start = (int)starts_data[0];
            int end = (int)ends_data[0];

            int last_dim = (int)data.m_shape.back();

            if (end == -1)
                end = last_dim;

            if (start < 0 || start > last_dim || end < 0 || end > last_dim || start >= end)
                throw std::invalid_argument(op.m_type + ": invalid value(s) in starts and/or ends.");

            std::vector<size_t> output_shape(data.m_shape);

            output_shape.back() = end - start;

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
            else
            {
                if (data.m_type != TensorDataType::float16) throw std::invalid_argument(op.m_type + ": wrong data type of data.");

                auto& data_data = data.get_vector<uint16_t>();

                tensor_vector<uint16_t> temp = create_tensor_vector<uint16_t>(output_num_els);
                output.set_vector(std::move(temp));

                sizeof_element = sizeof(uint16_t);
                ptr_data = data_data.data();
                ptr_output = output.get_vector<uint16_t>().data();
            }

            struct context
            {
                void* ptr_data, * ptr_output;
                size_t output_stride, data_stride, start, sizeof_element;
            };

            context prl_ctx;

            prl_ctx.start = start;

            prl_ctx.output_stride = output_shape.back();
            prl_ctx.data_stride = data.m_shape.back();

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

            if (!compare_shapes(output_shape, output.m_shape))
                throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

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

            if (!compare_shapes(output_shape, output.m_shape))
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

            if (q.m_shape[1] % m_attention_fused_ops_parts)
                throw std::invalid_argument(op.m_type + ": m_attention_fused_ops_parts is not valid.");

            std::vector<size_t> aux_shape({ q.m_shape[1] / m_attention_fused_ops_parts, k.m_shape[2] });

            size_t aux_num_els = 1;
            for (auto& s : aux_shape)
                aux_num_els *= s;

            tensor_vector<uint8_t> aux_0 = create_tensor_vector<uint8_t>(aux_num_els * sizeof_element);
            tensor_vector<uint8_t> aux_1 = create_tensor_vector<uint8_t>(aux_num_els * sizeof_element);

            std::vector<size_t> q_part_shape({ q.m_shape[1] / m_attention_fused_ops_parts, q.m_shape[2] });

            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < m_attention_fused_ops_parts; j++)
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

            if (!compare_shapes(output_shape, output.m_shape))
                throw std::invalid_argument(op.m_type + ": unexpected shape of output.");

            output.set_vector(std::move(output_data));
            push_tensor(std::move(output));
        }
        else
            throw std::invalid_argument("Operator not implemented: " + op.m_type + ".");
    }

    for (auto& t : m_data)
    {
        if (t.m_next_part)
            throw std::invalid_argument("Model::run: multipart tensors are intended to be used internally.");

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
        else if (t.m_type != TensorDataType::float32)
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

}
