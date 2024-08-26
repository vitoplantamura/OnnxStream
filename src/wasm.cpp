#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/threading.h>
#define ONNXSTREAM_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define ONNXSTREAM_EXPORT
#endif

#include "onnxstream.h"

using namespace onnxstream;

class ModelContext
{
public:

	Model m_model;
	std::string m_def;
};

ONNXSTREAM_EXPORT ModelContext* model_new()
{
	ModelContext* obj = new ModelContext();

	obj->m_model.set_weights_provider(RamWeightsProvider<WeightsProvider>());

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

ONNXSTREAM_EXPORT char* model_get_weights_names(ModelContext* obj)
{
	Model model;
	model.m_support_dynamic_shapes = true;
	model.set_weights_provider(CollectNamesWeightsProvider(/* use_vector */ true));
	model.read_string(obj->m_def.c_str());
	model.init();
	auto& names = model.get_weights_provider<CollectNamesWeightsProvider>().m_names_vec;

	std::string ret;

	for (auto& name : names)
	{
		auto fn = name;
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

ONNXSTREAM_EXPORT void* model_add_weights_file(ModelContext* obj, char* name, unsigned int size)
{
	return obj->m_model.get_weights_provider<RamWeightsProvider<WeightsProvider>>().add_empty_and_return_ptr<float>(name, size);
}

ONNXSTREAM_EXPORT void* model_add_tensor(ModelContext* obj, char* name, unsigned int dims_num, unsigned int* dims)
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

	tensor_vector<float> data(num_els);
	t.set_vector(std::move(data));
	obj->m_model.push_tensor(std::move(t));

	return obj->m_model.m_data.back().get_vector<float>().data();
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
	if (!t)
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
