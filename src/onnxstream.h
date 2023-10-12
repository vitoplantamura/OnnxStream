#pragma once

#include <limits>
#include <vector>
#include <functional>
#include <variant>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <any>
#include <map>
#include <string>
#include <fstream>
#include <optional>
#include <set>
#include <memory>

namespace onnxstream {

#define TENSOR_VECTOR_EXTRA_BYTES 16

template<class T> // from: https://en.cppreference.com/w/cpp/named_req/Allocator
struct Mallocator
{
    typedef T value_type;

    Mallocator() = default;

    template<class U>
    constexpr Mallocator(const Mallocator <U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        if (auto p = static_cast<T*>(std::malloc(n * sizeof(T) + TENSOR_VECTOR_EXTRA_BYTES)))
        {
            //report(p, n);
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t n) noexcept
    {
        //report(p, n, 0);
        std::free(p);
    }
    /*private:
        void report(T* p, std::size_t n, bool alloc = true) const
        {
            std::cout << (alloc ? "Alloc: " : "Dealloc: ") << sizeof(T) * n
                << " bytes at " << std::hex << std::showbase
                << reinterpret_cast<void*>(p) << std::dec << '\n';
        }*/
};

template<class T, class U>
bool operator==(const Mallocator <T>&, const Mallocator <U>&) { return true; }

template<class T, class U>
bool operator!=(const Mallocator <T>&, const Mallocator <U>&) { return false; }

template <class T>
using tensor_vector = std::vector<T, Mallocator<T>>;

template<typename T>
tensor_vector<T> create_tensor_vector(size_t size)
{
    return tensor_vector<T>(size); // todo
}

// ---

class scope_guard {
public:
    template<class Callable>
    scope_guard(Callable&& undo_func) : f(std::forward<Callable>(undo_func)) { }

    ~scope_guard() {
        if (f) f();
    }

    scope_guard() = delete;
    scope_guard(const scope_guard&) = delete;
    scope_guard& operator = (const scope_guard&) = delete;
    scope_guard(scope_guard&&) = delete;
    scope_guard& operator = (scope_guard&&) = delete;

private:
    std::function<void()> f;
};

// ---

template<typename T>
T read_file(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("read_file: unable to open file (" + std::string(filename) + ").");

    file.seekg(0, std::ios::end);
    std::streampos size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0 || (size % sizeof(typename T::value_type)))
        throw std::invalid_argument("read_file: invalid size of file.");

    T data(size / sizeof(typename T::value_type)); // todo

    file.read((char*)&data[0], size);

    if (!file)
        throw std::runtime_error("read_file: unable to read file.");

    return data;
}

template <typename T>
void write_file(const char* filename, const T& data)
{
    std::ofstream file(filename, std::ios::binary | std::ios::out);

    if (!file)
        throw std::runtime_error("write_file: unable to open file.");

    file.write((const char*)&data[0], data.size() * sizeof(typename T::value_type));

    if (!file)
        throw std::runtime_error("write_file: unable to write file.");
}

std::string& trim(std::string& s);

// ---

enum class TensorDataType
{
    none,
    uint8,
    float16,
    float32,
    int64
};

enum class TensorDataLayout
{
    unspecified,
    nhwc
};

class Tensor
{
public:

    std::string m_name;

    TensorDataType m_type = TensorDataType::none;

    std::vector<size_t> m_shape;

    std::variant<
        std::shared_ptr<tensor_vector<uint8_t>>,
        std::shared_ptr<tensor_vector<uint16_t>>,
        std::shared_ptr<tensor_vector<float>>,
        std::shared_ptr<tensor_vector<int64_t>>
    > m_data;

    TensorDataLayout m_layout = TensorDataLayout::unspecified;

    std::unique_ptr<Tensor> m_next_part;

    float m_scale = 0;
    uint8_t m_zero_point = 0;

public:

    template <typename T>
    tensor_vector<T>& get_vector()
    {
        switch (m_type)
        {
        case TensorDataType::uint8:
            if constexpr (!std::is_same<T, uint8_t>::value) throw std::invalid_argument("Tensor::get_vector: invalid type.");
            break;
        case TensorDataType::float16:
            if constexpr (!std::is_same<T, uint16_t>::value) throw std::invalid_argument("Tensor::get_vector: invalid type.");
            break;
        case TensorDataType::float32:
            if constexpr (!std::is_same<T, float>::value) throw std::invalid_argument("Tensor::get_vector: invalid type.");
            break;
        case TensorDataType::int64:
            if constexpr (!std::is_same<T, int64_t>::value) throw std::invalid_argument("Tensor::get_vector: invalid type.");
            break;
        default:
            throw std::invalid_argument("Tensor::get_vector: invalid type.");
        }

        return *std::get<std::shared_ptr<tensor_vector<T>>>(m_data);
    }

    template <typename T>
    void set_vector(tensor_vector<T>&& v)
    {
        if constexpr (std::is_same<T, uint8_t>::value)
            m_type = TensorDataType::uint8;
        else if constexpr (std::is_same<T, uint16_t>::value)
            m_type = TensorDataType::float16;
        else if constexpr (std::is_same<T, float>::value)
            m_type = TensorDataType::float32;
        else if constexpr (std::is_same<T, int64_t>::value)
            m_type = TensorDataType::int64;
        else
            throw std::invalid_argument("Tensor::set_vector: invalid type.");

        m_data = std::make_shared<tensor_vector<T>>(std::move(v));
    }

    Tensor get_copy_without_data()
    {
        Tensor ret;
        ret.m_name = m_name;
        ret.m_type = m_type;
        ret.m_shape = m_shape;
        ret.m_layout = m_layout;
        ret.m_scale = m_scale;
        ret.m_zero_point = m_zero_point;
        return ret;
    }

    void reset_data()
    {
        m_type = TensorDataType::none;
        m_shape.clear();
        m_data = decltype(m_data)();
        m_layout = TensorDataLayout::unspecified;
        m_scale = 0;
        m_zero_point = 0;
    }
};

class Operation
{
public:

    std::string m_name;
    std::string m_type;

    std::vector<Tensor> m_input;
    std::vector<Tensor> m_output;

    std::vector<std::pair<std::string, std::string>> m_attributes;
};

class WeightsProvider
{
public:

    WeightsProvider() {}
    virtual ~WeightsProvider() {}

    std::string m_path;

    virtual void on_init(TensorDataType type, const std::string& name, size_t size) {}
    virtual void on_restart() {}

    virtual tensor_vector<uint8_t> get_uint8(const std::string& name) = 0;
    virtual tensor_vector<uint16_t> get_float16(const std::string& name) = 0;
    virtual tensor_vector<float> get_float32(const std::string& name) = 0;
    virtual tensor_vector<int64_t> get_int64(const std::string& name) = 0;
};

class DiskNoCacheWeightsProvider : public WeightsProvider
{
public:

    virtual tensor_vector<uint8_t> get_uint8(const std::string& name)
    {
        return read_file<tensor_vector<uint8_t>>((m_path + name).c_str());
    }

    virtual tensor_vector<uint16_t> get_float16(const std::string& name)
    {
        return read_file<tensor_vector<uint16_t>>((m_path + name).c_str());
    }

    virtual tensor_vector<float> get_float32(const std::string& name)
    {
        return read_file<tensor_vector<float>>((m_path + name).c_str());
    }

    virtual tensor_vector<int64_t> get_int64(const std::string& name)
    {
        return read_file<tensor_vector<int64_t>>((m_path + name).c_str());
    }
};

class DiskPrefetchWeightsProvider : public WeightsProvider
{
private:

    size_t m_max_memory = 0;
    bool m_limit_plus_one_file = false;

private:

    struct Entry
    {
        TensorDataType m_type;
        std::string m_name;
        size_t m_size;

        std::variant<
            std::monostate,
            tensor_vector<uint8_t>,
            tensor_vector<uint16_t>,
            tensor_vector<float>,
            tensor_vector<int64_t>
        > m_data;

        Entry(TensorDataType type, const std::string& name, size_t size) :
            m_type(type), m_name(name), m_size(size) {}
        Entry() {}
    };

private:

    std::vector<Entry> m_entries, m_entries_copy;

    bool m_exit;
    std::string m_error;

    std::thread* m_thread;
    std::condition_variable* m_cv;
    std::mutex* m_mutex;

    void init_members()
    {
        m_entries = m_entries_copy;

        m_exit = false;
        m_error = "";

        m_thread = nullptr;
        m_cv = nullptr;
        m_mutex = nullptr;
    }

    void destructor()
    {
        if (m_thread)
        {
            {
                std::lock_guard lock(*m_mutex);
                m_exit = true;
            }

            m_cv->notify_one();

            m_thread->join();
        }

        if (m_cv)
            delete m_cv;
        if (m_mutex)
            delete m_mutex;
        if (m_thread)
            delete m_thread;
    }

private:

    void worker()
    {
        std::string path = m_path;
        std::string error;

        try
        {
            if (!m_max_memory)
                throw std::invalid_argument("DiskPrefetchWeightsProvider::worker: m_max_memory cannot be zero.");

            while (true)
            {
                int index = -1;
                Entry entry;

                {
                    bool exit = false;

                    std::unique_lock lock(*m_mutex);
                    m_cv->wait(lock, [&]() {

                        if (m_exit || !m_entries.size() ||
                            !std::holds_alternative<std::monostate>(m_entries[0].m_data))
                        {
                            exit = true;
                            return true;
                        }

                        size_t memory = 0;
                        bool for_exit_next_iter = false;

                        for (int i = m_entries.size() - 1; i >= 0; i--)
                        {
                            auto& e = m_entries[i];

                            if (!std::holds_alternative<std::monostate>(e.m_data))
                            {
                                memory += e.m_size;
                                if (memory >= m_max_memory)
                                    if (!m_limit_plus_one_file || for_exit_next_iter)
                                        return false;
                                    else
                                        for_exit_next_iter = true;
                            }
                            else
                            {
                                index = i;
                                entry = e;
                                return true;
                            }
                        }

                        return false;
                    });

                    lock.unlock();

                    if (exit || index == -1)
                        break;
                }

                switch (entry.m_type)
                {
                case TensorDataType::uint8:
                    entry.m_data = std::move(read_file<tensor_vector<uint8_t>>((path + entry.m_name).c_str()));
                    break;
                case TensorDataType::float16:
                    entry.m_data = std::move(read_file<tensor_vector<uint16_t>>((path + entry.m_name).c_str()));
                    break;
                case TensorDataType::float32:
                    entry.m_data = std::move(read_file<tensor_vector<float>>((path + entry.m_name).c_str()));
                    break;
                case TensorDataType::int64:
                    entry.m_data = std::move(read_file<tensor_vector<int64_t>>((path + entry.m_name).c_str()));
                    break;
                default:
                    throw std::invalid_argument("DiskPrefetchWeightsProvider::worker: invalid type.");
                }

                {
                    std::lock_guard lock(*m_mutex);

                    if (index >= m_entries.size() ||
                        m_entries[index].m_name != entry.m_name ||
                        !std::holds_alternative<std::monostate>(m_entries[index].m_data))
                    {
                        throw std::invalid_argument("DiskPrefetchWeightsProvider::worker: inconsistent state of m_entries.");
                    }

                    m_entries[index].m_data = std::move(entry.m_data);
                }

                m_cv->notify_one();
            }

            return;
        }
        catch (const std::exception& e)
        {
            error = e.what();
        }

        {
            std::lock_guard lock(*m_mutex);
            m_error = error;
        }
    }

    template <typename T>
    tensor_vector<T> provide(const std::string& name)
    {
        if (!m_thread)
        {
            if (m_entries_copy.size() == 0)
            {
                std::reverse(m_entries.begin(), m_entries.end());
                m_entries_copy = m_entries;
            }

            m_cv = new std::condition_variable();
            m_mutex = new std::mutex();

            m_thread = new std::thread(&DiskPrefetchWeightsProvider::worker, this);
        }

        while (true)
        {
            tensor_vector<T> v;

            {
                std::lock_guard lock(*m_mutex);

                if (m_error.size())
                    throw std::invalid_argument("DiskPrefetchWeightsProvider::provide: fatal error in worker thread: \"" + m_error + "\".");

                if (!m_entries.size())
                    throw std::invalid_argument("DiskPrefetchWeightsProvider::provide: vector is empty.");

                auto& e = m_entries.back();

                if (e.m_name != name)
                    throw std::invalid_argument("DiskPrefetchWeightsProvider::provide: invalid name.");

                if (!std::holds_alternative<std::monostate>(e.m_data))
                {
                    if (!std::holds_alternative<tensor_vector<T>>(e.m_data))
                        throw std::invalid_argument("DiskPrefetchWeightsProvider::provide: invalid type.");

                    v = std::move(std::get<tensor_vector<T>>(e.m_data));

                    if (!v.size())
                        throw std::invalid_argument("DiskPrefetchWeightsProvider::provide: invalid size.");

                    m_entries.pop_back();
                }
            }

            m_cv->notify_one();

            if (v.size())
                return v;

            {
                using namespace std::chrono_literals;

                std::unique_lock lock(*m_mutex);
                m_cv->wait_for(lock, 33ms);
            }
        }
    }

public:

    DiskPrefetchWeightsProvider(size_t max_memory = 1 * 1024 * 1024, bool limit_plus_one_file = true)
        : m_max_memory(max_memory), m_limit_plus_one_file(limit_plus_one_file)
    {
        init_members();
    }

    virtual ~DiskPrefetchWeightsProvider()
    {
        destructor();
    }

    virtual void on_init(TensorDataType type, const std::string& name, size_t size)
    {
        std::string n = name;

        auto pos = n.find("_nchw.bin");
        if (pos != std::string::npos)
            n = n.substr(0, pos) + "_nhwc.bin";

        m_entries.emplace_back(type, n, size);
    }

    virtual void on_restart()
    {
        destructor();
        init_members();
    }

    virtual tensor_vector<uint8_t> get_uint8(const std::string& name)
    {
        return provide<uint8_t>(name);
    }

    virtual tensor_vector<uint16_t> get_float16(const std::string& name)
    {
        return provide<uint16_t>(name);
    }

    virtual tensor_vector<float> get_float32(const std::string& name)
    {
        return provide<float>(name);
    }

    virtual tensor_vector<int64_t> get_int64(const std::string& name)
    {
        return provide<int64_t>(name);
    }
};

template <typename T>
class RamWeightsProvider : public WeightsProvider
{
private:

    std::shared_ptr<T> m_reader;

private:

    struct Entry
    {
        std::string m_name;

        std::variant<
            std::monostate,
            tensor_vector<uint8_t>,
            tensor_vector<uint16_t>,
            tensor_vector<float>,
            tensor_vector<int64_t>
        > m_data;
    };

private:

    std::vector<Entry> m_weights;
    size_t m_index = 0;

    bool m_path_set = false;

private:

    template <typename U>
    tensor_vector<U> provide(const std::string& name)
    {
        if (m_reader)
        {
            if (!m_path_set)
            {
                m_reader->m_path = m_path;
                m_path_set = true;
            }

            tensor_vector<U> v;

            if constexpr (std::is_same<U, uint8_t>::value)
                v = m_reader->get_uint8(name);
            else if constexpr (std::is_same<U, uint16_t>::value)
                v = m_reader->get_float16(name);
            else if constexpr (std::is_same<U, float>::value)
                v = m_reader->get_float32(name);
            else if constexpr (std::is_same<U, int64_t>::value)
                v = m_reader->get_int64(name);
            else
                throw std::invalid_argument("RamWeightsProvider::provide: invalid type.");

            Entry e;
            e.m_name = name;
            e.m_data = v;
            m_weights.push_back(std::move(e));

            return v;
        }
        else
        {
            if (m_index >= m_weights.size())
                throw std::invalid_argument("RamWeightsProvider::provide: invalid index.");

            Entry& e = m_weights[m_index++];

            if (name != e.m_name)
                throw std::invalid_argument("RamWeightsProvider::provide: invalid name.");

            if (!std::holds_alternative<tensor_vector<U>>(e.m_data))
                throw std::invalid_argument("RamWeightsProvider::provide: invalid data type.");

            return std::get<tensor_vector<U>>(e.m_data);
        }
    }

public:

    RamWeightsProvider(T&& reader)
    {
        m_reader = std::make_shared<T>(std::move(reader));
    }

    virtual ~RamWeightsProvider()
    {
    }

    virtual void on_init(TensorDataType type, const std::string& name, size_t size)
    {
        if (m_reader)
            m_reader->on_init(type, name, size);
        else
            throw std::invalid_argument("RamWeightsProvider::on_init: invalid call to on_init.");
    }

    virtual void on_restart()
    {
        m_reader.reset();
        m_index = 0;
    }

    virtual tensor_vector<uint8_t> get_uint8(const std::string& name)
    {
        return provide<uint8_t>(name);
    }

    virtual tensor_vector<uint16_t> get_float16(const std::string& name)
    {
        return provide<uint16_t>(name);
    }

    virtual tensor_vector<float> get_float32(const std::string& name)
    {
        return provide<float>(name);
    }

    virtual tensor_vector<int64_t> get_int64(const std::string& name)
    {
        return provide<int64_t>(name);
    }
};

class XnnPack;

class Model
{
public:

    Model();
    ~Model();

    template <typename T>
    void set_weights_provider(T&& wp)
    {
        if (m_wp_interface_internal) throw std::invalid_argument("Model::set_weights_provider: weights provider already set.");
        m_wp_object = std::move(wp);
        m_wp_interface_internal = std::any_cast<T>(&m_wp_object);
    }

    void read_file(const char* filename);

    std::vector<Tensor> m_data;

    void push_tensor(Tensor&& t, bool force_quantization = false);

    void run();

    void read_range_data(const char* filename);
    void write_range_data(const char* filename);
    std::map<std::string, std::pair<float, float>> m_range_data;
    bool m_range_data_calibrate = false;

    bool m_use_fp16_arithmetic = false;
    bool m_use_uint8_qdq = false;
    bool m_use_uint8_arithmetic = false;
    bool m_pow_requires_float = false;
    bool m_do_multipart_quantization = false;
    size_t m_multipart_threshold = -1;
    bool m_fuse_ops_in_attention = false;
    size_t m_attention_fused_ops_parts = 2;
    std::vector<std::string> m_extra_outputs;
    bool m_force_fp16_storage = false;
    std::set<std::string> m_force_uint8_storage_set;

    bool m_ops_printf = false;

private:

    std::any m_wp_object;
    WeightsProvider* m_wp_interface_internal = nullptr;
    WeightsProvider* get_wp()
    {
        if (!m_wp_interface_internal) set_weights_provider(DiskPrefetchWeightsProvider());
        return m_wp_interface_internal;
    }

    static const size_t m_perthread_buffer_size;
    static const size_t m_float16_buffer_size;
    static const size_t m_float32_buffer_size;
    static const size_t m_float32_buffer_size_w_extra_bytes;

    std::string next_line();

    Tensor parse_tensor_string(std::string& str);

    std::optional<Operation> next_op();

    Tensor& get_tensor_data(Tensor& tensor, bool make_copy = false, bool requires_float = false, TensorDataLayout required_layout = TensorDataLayout::unspecified, bool accepts_multipart = false);

    static bool compare_shapes(const std::vector<size_t>& shape_1, const std::vector<size_t>& shape_2, int except = -1);

    std::vector<char> m_model;
    size_t m_pos = 0;

    std::string m_path;

    std::map<std::string, int> m_intermediate_refs, m_intermediate_refs_copy;

    std::vector<Operation> m_ops_queue;

    Tensor& get_multipart_input(Tensor& t, size_t i, TensorDataType base_type);
    void push_multipart_tensor(Tensor& output, bool is_multipart);
    size_t get_multipart_dimension(Tensor& t);

    static bool get_start_and_end(size_t& start, size_t& end, const size_t i, const size_t size, const size_t threads_count);

    std::optional<std::pair<float, float>> get_percentiles(Tensor& input, float from_left, float from_right);
    bool quantize(Tensor& input, float from_left, float from_right);
    void dequantize(Tensor& input, TensorDataType dest_type);
    std::pair<float, uint8_t> range_to_scale(std::pair<float, float> range);

    int m_ops_printf_index = 0;

    XnnPack* m_xnnpack = nullptr;
};

}
