#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <filesystem>

#include "onnxstream.h"
using namespace onnxstream;

struct MainArgs
{
    std::string m_path_with_slash = "./";
    bool m_ops_printf = false;
    bool m_mistral = false;
    bool m_download = false;
    std::string m_curl_parallel = "4";
    std::string m_cuda_vram_to_use = "0";
    bool m_cuda_compute_fp32 = false;
    bool m_ops_times_printf = false;
    bool m_no_fp16 = false;
};

static MainArgs g_main_args;

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

#ifdef _WIN32

    {
        HANDLE handleOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD consoleMode;
        GetConsoleMode(handleOut, &consoleMode);
        consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        consoleMode |= DISABLE_NEWLINE_AUTO_RETURN;
        SetConsoleMode(handleOut, consoleMode);
    }

#endif

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        std::string* str = nullptr;

        if (arg == "--models-path")
        {
            str = &g_main_args.m_path_with_slash;
        }
        else if (arg == "--ops-printf")
        {
            g_main_args.m_ops_printf = true;
        }
        else if (arg == "--mistral")
        {
            g_main_args.m_mistral = true;
        }
        else if (arg == "--download")
        {
            g_main_args.m_download = true;
        }
        else if (arg == "--curl-parallel")
        {
            str = &g_main_args.m_curl_parallel;
        }
        else if (arg == "--ops-times")
        {
            g_main_args.m_ops_times_printf = true;
        }
        else if (arg == "--no-fp16")
        {
            g_main_args.m_no_fp16 = true;
        }
#if ONNXSTREAM_CUDA
        else if (arg == "--cuda")
        {
            str = &g_main_args.m_cuda_vram_to_use;
        }
        else if (arg == "--cuda-fp32")
        {
            g_main_args.m_cuda_compute_fp32 = true;
        }
#endif
        else
        {
            printf(("Invalid command line argument: \"" + arg + "\".\n\n").c_str());

            printf("--mistral           Runs Mistral 7B instead of TinyLlama 1.1B.\n");
            printf("--models-path       Sets the base folder containing the models. Default is \".\".\n");
            printf("--no-fp16           Do not use FP16 arithmetic.\n");
#if ONNXSTREAM_CUDA
            printf("--cuda              Enables partial GPU offloading. Requires the num of GB to offload.\n");
            printf("--cuda-fp32         Forces FP32 compute with Cublas.\n");
#endif
            printf("--ops-printf        During inference, writes the current operation to stdout.\n");
            printf("--ops-times         During inference, writes the operators' execution times to stdout.\n");
            printf("--download          Forces the (re)download of the current model.\n");
            printf("--curl-parallel     Sets the number of parallel downloads with CURL. Default is 4.\n");

            return -1;
        }

        if (str)
        {
            if (++i >= argc)
            {
                printf(("Argument \"" + arg + "\" should be followed by a string.").c_str());
                return -1;
            }

            *str = argv[i];
        }
    }

    trim(g_main_args.m_path_with_slash);
    if (g_main_args.m_path_with_slash.size() &&
        g_main_args.m_path_with_slash.find_last_of("/\\") != g_main_args.m_path_with_slash.size() - 1)
    {
        g_main_args.m_path_with_slash += "/";
    }

    try
    {
        int curl_parallel = std::stoi(g_main_args.m_curl_parallel);
        if (curl_parallel < 1 || curl_parallel > 128)
            throw std::invalid_argument("--curl-parallel must be between 1 and 128.");

        double cuda_vram_to_use = std::stod(g_main_args.m_cuda_vram_to_use);

        bool is_tiny = !g_main_args.m_mistral;

        std::string name = is_tiny ? "TinyLlama-1.1B-Chat-v0.3-fp16" : "Mistral-7B-Instruct-v0.2-fp16";

        printf("Model: %s\n", name.c_str());

        std::string folder_with_slash = g_main_args.m_path_with_slash + name + "/";
        std::string url_with_slash = "https://huggingface.co/vitoplantamura/onnxstream-llms/resolve/main/" + name + "/";

        FILE* model_txt_test = nullptr;

        if (!g_main_args.m_download)
        {
            model_txt_test = fopen((folder_with_slash + "model.txt").c_str(), "r");
            if (model_txt_test)
                fclose(model_txt_test);
        }

        auto download_file = [](const std::vector<std::pair<std::string, std::string>>& urls) -> void
        {
#ifdef _WIN32
            std::string null_device = "NUL";
#else
            std::string null_device = "/dev/null";
#endif
            std::string command = "curl --location --fail --silent --show-error --parallel ";

            for (const auto& url : urls)
                command += " -o \"" + url.second + "\" \"" + url.first + "\" ";
            command += " >" + null_device + " 2>&1";

            if (system(command.c_str()))
                throw std::invalid_argument("Download error.");
        };

        if (!model_txt_test)
        {
            printf("Downloading model.txt and vocab.txt...");

            std::filesystem::create_directory(folder_with_slash);

            std::vector<std::pair<std::string, std::string>> txt_files = {
                std::make_pair(url_with_slash + "model.txt", folder_with_slash + "model.txt"),
                std::make_pair(url_with_slash + "vocab.txt", folder_with_slash + "vocab.txt")
            };

            download_file(txt_files);

            printf(" done!\n");

            Model model;

            model.m_support_dynamic_shapes = true;
            model.set_weights_provider(CollectNamesWeightsProvider());

            model.read_file((folder_with_slash + "model.txt").c_str());

            model.init();

            auto& names = model.get_weights_provider<CollectNamesWeightsProvider>().m_names;

            int counter = 0;
            std::vector<std::pair<std::string, std::string>> bin_files;
            for (auto& name : names)
            {
                printf("\rDownloading weights: %i/%i...", ++counter, (int)names.size());
                fflush(stdout);
                bin_files.emplace_back(url_with_slash + name, folder_with_slash + name);
                if (counter % curl_parallel == 0 || counter == (int)names.size())
                {
                    download_file(bin_files);
                    bin_files.clear();
                }
            }

            printf(" done!\n");
        }

        std::unordered_map<std::string, int> token2idx;
        std::vector<std::pair<int, std::string>> idx2token;

        std::string pathname = folder_with_slash + "vocab.txt";
        std::ifstream infile;
        infile.open(pathname);
        if (!infile)
            throw std::invalid_argument("unable to open file: " + pathname);

        auto add_token = [&](const std::string& token, int score)
        {
            token2idx[token] = idx2token.size();
            idx2token.emplace_back(score, token);
        };

        {
            std::string s;
            while (getline(infile, s))
            {
                size_t comma = s.find(',');
                if (comma == std::string::npos)
                    throw std::invalid_argument("invalid format of tokenizer file's line.");

                std::string token = s.substr(comma + 1);
                int score = std::stoi(s.substr(0, comma));

                if (token.size() == 6 &&
                    token.substr(0, 3) == "<0x" &&
                    token[5] == '>')
                {
                    char c = (char)(std::stoul(token.substr(3, 2), nullptr, 16) & 0xFF);
                    token = std::string(1, c);
                }

                add_token(token, score);
            }
            infile.close();
        }

        std::vector<size_t> special_toks;

        if (is_tiny)
        {
            size_t special_toks_start = idx2token.size();

            add_token("[PAD]", 0);
            add_token("<|im_start|>", 0);
            add_token("<|im_end|>", 0);

            for (size_t j = special_toks_start; j < idx2token.size(); j++)
                special_toks.push_back(j);
        }

        auto add_special_tok = [&](const std::string& token)
        {
            auto it = token2idx.find(token);
            if (it == token2idx.end())
                throw std::invalid_argument("Special token not found.");
            special_toks.push_back(it->second);
        };

        add_special_tok("<s>");
        add_special_tok("</s>");

        auto encode = [&](const std::string& s) -> tensor_vector<int64_t>
        {
            tensor_vector<int64_t> r;

            for (size_t i = 0; i < s.size(); i++)
            {
                bool found = false;
                for (size_t j : special_toks)
                {
                    auto& t = idx2token[j];
                    if (s.substr(i, t.second.size()) == t.second)
                    {
                        r.push_back(j);
                        i += t.second.size() - 1;
                        found = true;
                        break;
                    }
                }
                if (found) continue;

                auto it = token2idx.find(std::string(1, s[i]));
                if (it == token2idx.end())
                    throw std::invalid_argument("Character not found (UNICODE not implemented yet).");
                r.push_back(it->second);
            }

            while (true)
            {
                int sc = std::numeric_limits<int>::min();
                int c = -1;
                int k = -1;

                for (int i = 0; i < (int)r.size() - 1; i++)
                {
                    auto it = token2idx.find(idx2token[r[i]].second + idx2token[r[i + 1]].second);
                    if (it != token2idx.end() && idx2token[it->second].first > sc)
                    {
                        sc = idx2token[it->second].first;
                        c = it->second;
                        k = i;
                    }
                }

                if (c == -1 || k == -1)
                    break;
                else
                {
                    r[k] = c;
                    r.erase(r.begin() + k + 1);
                }
            }

            return r;
        };

        auto get_output = [](std::vector<Tensor>& data, const std::string& name) -> Tensor {

            for (size_t i = 0; i < data.size(); i++)
                if (data[i].m_name == name)
                {
                    Tensor t = std::move(data[i]);
                    data.erase(data.begin() + i);
                    return t;
                }

            throw std::invalid_argument("output not found.");
        };

        auto argmax = [](Tensor& res) -> int
        {
            auto& vec = res.get_vector<float>();

            float prev = -1;
            int idx = -1;
            int base = res.m_shape[2] * (res.m_shape[1] - 1);
            for (int k = 0; k < res.m_shape[2]; k++)
                if (vec[base + k] >= prev)
                {
                    prev = vec[base + k];
                    idx = k;
                }

            return idx;
        };

        Model model;

        model.m_ops_printf = g_main_args.m_ops_printf;
        model.m_ops_times_printf = g_main_args.m_ops_times_printf;
        model.m_support_dynamic_shapes = true;
        model.m_use_fp16_arithmetic = !g_main_args.m_no_fp16;
        model.m_use_ops_cache = true;
        model.m_use_scaled_dp_attn_op = true;
        model.m_outputs_convert_set = { "logits" };
        model.m_use_next_op_cache = true;
        model.set_weights_provider(RamWeightsProvider<DiskPrefetchWeightsProvider>(DiskPrefetchWeightsProvider()));
        model.set_cuda_options(CudaOptions((uint64_t)(cuda_vram_to_use * 1024 * 1024 * 1024), g_main_args.m_cuda_compute_fp32));

        model.m_requires_upcast = [](const std::string& op_type, const std::string& op_name) -> bool
        {
            return op_name.find("/input_layernorm/") != std::string::npos ||
                op_name.find("/post_attention_layernorm/") != std::string::npos;
        };

        for (int i = 0; i < (!is_tiny ? 64 : 44); i++)
            model.m_extra_outputs.push_back("opkv" + std::to_string(i));

        model.read_file((folder_with_slash + "model.txt").c_str());

        auto forward = [&](tensor_vector<int64_t>& input_ids, tensor_vector<int64_t>& position_ids, tensor_vector<int64_t>& attention_mask)
        {
            if (!model.m_data.size())
            {
                for (int k = 0; k < (!is_tiny ? 64 : 44); k++)
                {
                    tensor_vector<float> pkv_data(0, 0);

                    Tensor t4;
                    t4.m_name = "pkv" + std::to_string(k);
                    if (!is_tiny)
                        t4.m_shape = { 1, 8, 0, 128 };
                    else
                        t4.m_shape = { 1, 4, 0, 64 };
                    t4.set_vector(std::move(pkv_data));
                    model.push_tensor(std::move(t4));
                }
            }
            else
            {
                for (auto& t : model.m_data)
                    if (t.m_name.find("pkv") == 1)
                        t.m_name.erase(0, 1);
            }

            Tensor t1;
            t1.m_name = "input_5F_ids";
            t1.m_shape = { 1, input_ids.size() };
            t1.set_vector(std::move(input_ids));
            model.push_tensor(std::move(t1));

            Tensor t2;
            t2.m_name = "position_5F_ids";
            t2.m_shape = { 1, position_ids.size() };
            t2.set_vector(std::move(position_ids));
            model.push_tensor(std::move(t2));

            Tensor t3;
            t3.m_name = "attention_5F_mask";
            t3.m_shape = { 1, attention_mask.size() };
            t3.set_vector(std::move(attention_mask));
            model.push_tensor(std::move(t3));

            model.run();
        };

        {
            printf("Loading weights...");

            tensor_vector<int64_t> input_ids(1, 0);
            tensor_vector<int64_t> position_ids(1, 0);
            tensor_vector<int64_t> attention_mask(1, 1);

            forward(input_ids, position_ids, attention_mask);

            model.m_data.clear();

            printf(" done!\n");
        }

        tensor_vector<int64_t> toks;

        while (true)
        {
            printf("\n>>> ");

            std::string prompt;
            std::getline(std::cin, prompt);

            std::string str = !is_tiny ?
                (std::string(toks.size() ? "</s>" : "<s>") + "[INST] " + prompt + " [/INST]") :
                (std::string(toks.size() ? "<|im_end|>\n" : "") + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n");

            auto new_toks = encode(str);
            toks.insert(toks.end(), new_toks.begin(), new_toks.end());

            for (int pos = 0; ; pos++)
            {
                tensor_vector<int64_t> position_ids;
                for (size_t k = toks.size() - new_toks.size(); k < toks.size(); k++)
                    position_ids.push_back((int64_t)k);

                tensor_vector<int64_t> attention_mask(position_ids.back() + 1, 1);

                forward(new_toks, position_ids, attention_mask);

                Tensor res = std::move(get_output(model.m_data, "logits"));
                int idx = argmax(res);
                if (idx < 0 || idx >= idx2token.size())
                    throw std::invalid_argument("Invalid result of the Argmax.");

                auto& tok = idx2token[idx].second;

                if ((is_tiny && tok == "<|im_end|>") || (!is_tiny && tok == "</s>"))
                    break;
                else if (tok.size())
                    printf("%s", tok.substr(pos == 0 && tok[0] == ' ' ? 1 : 0).c_str());

                toks.push_back(idx);
                new_toks = { idx };
            }
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        printf("=== ERROR === %s\n", e.what());
        return -1;
    }

    return 0;
}
