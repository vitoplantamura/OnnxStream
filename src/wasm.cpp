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
	return new ModelContext();
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
    model.set_weights_provider(CollectNamesWeightsProvider());
    model.read_string(obj->m_def.c_str());
    model.init();
    auto& names = model.get_weights_provider<CollectNamesWeightsProvider>().m_names;

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
