#ifdef __EMSCRIPTEN__
  #include <emscripten.h>
  #include <emscripten/threading.h>
  #define ONNXSTREAM_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#elif defined(_WIN32)
  #define ONNXSTREAM_EXPORT extern "C" __declspec(dllexport)
#else
  #define ONNXSTREAM_EXPORT extern "C"
#endif

#include "onnxstream.h"
#include <cstring>

using namespace onnxstream;

static char* pchar_to_buf(const char* pchar)
{
	char* buf = (char*)::malloc(::strlen(pchar) + 1);
	::strcpy(buf, pchar);
	return buf;
}

static char* except_to_err(const std::exception& e)
{
	return pchar_to_buf(e.what());
}

class ModelContext
{
public:

	ModelContext(int threads_count)
		: m_model(threads_count)
	{
	}

	Model m_model;
	std::string m_def;
	std::string m_wp;
};

ONNXSTREAM_EXPORT ModelContext* model_new()
{
	int threads_count = 0;

#ifdef __EMSCRIPTEN_PTHREADS__
	threads_count = emscripten_num_logical_cores() / 2;
	if (threads_count < 1)
		threads_count = 1;
	else if (threads_count > 8)
		threads_count = 8;
#endif

	ModelContext* obj = new ModelContext(threads_count);

	obj->m_wp = "ram";
	obj->m_model.set_weights_provider(RamWeightsProvider<WeightsProvider>());

	return obj;
}

ONNXSTREAM_EXPORT ModelContext* model_new_2(int threads_count, char* wp_name)
{
	ModelContext* obj = new ModelContext(threads_count);

	obj->m_wp = wp_name;

	if (!::strcmp(wp_name, "ram"))
		obj->m_model.set_weights_provider(RamWeightsProvider<WeightsProvider>());
	else if (!::strcmp(wp_name, "nocache"))
		obj->m_model.set_weights_provider(DiskNoCacheWeightsProvider());
	else if (!::strcmp(wp_name, "prefetch"))
		obj->m_model.set_weights_provider(DiskPrefetchWeightsProvider());
	else if (!::strcmp(wp_name, "ram+nocache"))
		obj->m_model.set_weights_provider(RamWeightsProvider<DiskNoCacheWeightsProvider>(DiskNoCacheWeightsProvider()));
	else if (!::strcmp(wp_name, "ram+prefetch"))
		obj->m_model.set_weights_provider(RamWeightsProvider<DiskPrefetchWeightsProvider>(DiskPrefetchWeightsProvider()));
	else
	{
		delete obj;
		return nullptr;
	}

	return obj;
}

ONNXSTREAM_EXPORT void model_delete(ModelContext* obj)
{
	delete obj;
}

ONNXSTREAM_EXPORT void model_read_string(ModelContext* obj, char* str)
{
	obj->m_def = str;
	obj->m_model.read_string(str);
}

ONNXSTREAM_EXPORT char* model_read_file(ModelContext* obj, char* fn)
{
	try
	{
		obj->m_model.read_file(fn);
		return nullptr;
	}
	catch (const std::exception& e)
	{
		return except_to_err(e);
	}
}

ONNXSTREAM_EXPORT char* model_get_weights_names(ModelContext* obj)
{
	Model model(/* threads_count */ -1);
	model.m_support_dynamic_shapes = true;
	model.set_weights_provider(CollectNamesWeightsProvider(/* use_vector */ true));
	model.read_string(obj->m_def.c_str());
	model.init();
	auto& names = model.get_weights_provider<CollectNamesWeightsProvider>().m_names_vec;

	std::string ret;

	for (auto& name : names)
	{
		if (name.m_type == TensorDataType::uint8)
			ret += "uint8:";
		else if (name.m_type == TensorDataType::float16)
			ret += "float16:";
		else if (name.m_type == TensorDataType::float32)
			ret += "float32:";
		else if (name.m_type == TensorDataType::int64)
			ret += "int64:";
		else
			throw std::invalid_argument("Unsupported tensor data format.");

		auto fn = name.m_name;
		auto lpos = fn.find("_nchw.bin");
		if (lpos != std::string::npos)
			fn = fn.substr(0, lpos) + "_nhwc.bin";
		ret += fn + "|";
	}

	if (ret.size())
		ret.pop_back();

	char* c_ret = (char*)::malloc(ret.size() + 1);
	::memcpy(c_ret, ret.c_str(), ret.size() + 1);
	return c_ret;
}

ONNXSTREAM_EXPORT void* model_add_weights_file(ModelContext* obj, char* type, char* name, unsigned int size)
{
	if (obj->m_wp != "ram")
		return nullptr;

	auto& wp = obj->m_model.get_weights_provider<RamWeightsProvider<WeightsProvider>>();

	if (!::strcmp(type, "uint8"))
		return wp.add_empty_and_return_ptr<uint8_t>(name, size / sizeof(uint8_t));
	else if (!::strcmp(type, "float16"))
		return wp.add_empty_and_return_ptr<uint16_t>(name, size / sizeof(uint16_t));
	else if (!::strcmp(type, "float32"))
		return wp.add_empty_and_return_ptr<float>(name, size / sizeof(float));
	else if (!::strcmp(type, "int64"))
		return wp.add_empty_and_return_ptr<int64_t>(name, size / sizeof(int64_t));
	else
		throw std::invalid_argument("Unsupported tensor data format.");
}

ONNXSTREAM_EXPORT void* model_add_tensor(ModelContext* obj, char* type, char* name, unsigned int dims_num, unsigned int* dims)
{
	Tensor t;
	t.m_name = name;

	size_t num_els = 1;
	t.m_shape.reserve(dims_num);
	for (size_t i = 0; i < dims_num; i++)
	{
		t.m_shape.push_back(dims[i]);
		num_els *= dims[i];
	}

	if (!::strcmp(type, "float32"))
	{
		tensor_vector<float> data(num_els);
		t.set_vector(std::move(data));
	}
	else if (!::strcmp(type, "int64"))
	{
		tensor_vector<int64_t> data(num_els);
		t.set_vector(std::move(data));
	}
	else
		throw std::invalid_argument("Unsupported tensor data format.");

	obj->m_model.push_tensor(std::move(t));

	if (!::strcmp(type, "float32"))
		return obj->m_model.m_data.back().get_vector<float>().data();
	else if (!::strcmp(type, "int64"))
		return obj->m_model.m_data.back().get_vector<int64_t>().data();
	else
		throw std::invalid_argument("Unsupported tensor data format.");
}

ONNXSTREAM_EXPORT void* model_get_tensor(ModelContext* obj, char* name)
{
	Tensor* t = nullptr;
	for (auto& tensor : obj->m_model.m_data)
		if (tensor.m_name == name)
		{
			t = &tensor;
			break;
		}
	if (!t || t->m_type != TensorDataType::float32)
		return nullptr;

	struct ReturnLayout
	{
		size_t dims_num;
		size_t* dims;
		size_t data_num;
		float* data;
	};

	ReturnLayout* c_ret = (ReturnLayout*)::malloc(sizeof(ReturnLayout));

	c_ret->dims_num = t->m_shape.size();
	c_ret->dims = t->m_shape.data();
	c_ret->data_num = t->get_vector<float>().size();
	c_ret->data = t->get_vector<float>().data();

	return c_ret;
}

ONNXSTREAM_EXPORT char* model_get_all_tensor_names(ModelContext* obj)
{
	std::string ret;
	for (auto& tensor : obj->m_model.m_data)
		ret += tensor.m_name + "|";
	if (ret.size())
		ret.pop_back();
	return pchar_to_buf(ret.c_str());
}

ONNXSTREAM_EXPORT void model_run(ModelContext* obj)
{
	try
	{
		obj->m_model.run();
	}
	catch (const std::exception& e)
	{
		printf("=== ERROR === %s\n", e.what());
		throw;
	}
}

ONNXSTREAM_EXPORT char* model_run_2(ModelContext* obj)
{
	try
	{
		obj->m_model.run();
		return nullptr;
	}
	catch (const std::exception& e)
	{
		return except_to_err(e);
	}
}

ONNXSTREAM_EXPORT void model_clear_tensors(ModelContext* obj)
{
	obj->m_model.m_data.clear();
}

ONNXSTREAM_EXPORT void model_set_option(ModelContext* obj, char* name, unsigned int value)
{
	bool set = false;

#define DEFINE_OPTION_BOOL(O) if (!::strcmp(name, #O)) { obj->m_model.m_##O = value ? true : false; set = true; }
	DEFINE_OPTION_BOOL(use_fp16_arithmetic)
	DEFINE_OPTION_BOOL(use_uint8_qdq)
	DEFINE_OPTION_BOOL(use_uint8_arithmetic)
	DEFINE_OPTION_BOOL(fuse_ops_in_attention)
	DEFINE_OPTION_BOOL(force_fp16_storage)
	DEFINE_OPTION_BOOL(support_dynamic_shapes)
	DEFINE_OPTION_BOOL(use_ops_cache)
	DEFINE_OPTION_BOOL(use_scaled_dp_attn_op)
	DEFINE_OPTION_BOOL(use_next_op_cache)
	DEFINE_OPTION_BOOL(ops_printf)
	DEFINE_OPTION_BOOL(ops_times_printf)
	DEFINE_OPTION_BOOL(use_nchw_convs)
#undef DEFINE_OPTION_BOOL

	if (!set)
	{
		const char* err = "model_set_option: 'name' not found.";
		printf("=== ERROR === %s\n", err);
		throw std::invalid_argument(err);
	}
}

ONNXSTREAM_EXPORT void model_add_extra_output(ModelContext* obj, char* name)
{
	obj->m_model.m_extra_outputs.emplace_back(name);
}

ONNXSTREAM_EXPORT void model_free_buffer(void* ptr)
{
	::free(ptr);
}
