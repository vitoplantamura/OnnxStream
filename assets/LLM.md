January 14, 2024:

# OnnxStream running TinyLlama 1.1B and Mistral 7B with initial GPU support

**NOTE:** if you use **Windows** and want to try TinyLlama 1.1B and Mistral 7B with OnnxStream, you can download the EXE file from the [Releases](https://github.com/vitoplantamura/OnnxStream/releases) of this repo. The EXE file will automatically download the selected model's parameters from [Hugging Face](https://huggingface.co/vitoplantamura/onnxstream-llms/tree/main) on the first execution (no need to git clone). For the **other supported platforms**, you need to build the application following the [instructions](https://github.com/vitoplantamura/OnnxStream#how-to-build-the-stable-diffusion-example-on-linuxmacwindowstermux) in the main README, specifying the "-DOS_LLM=ON" option in cmake, and, optionally, the "-DOS_CUDA=ON" option to enable GPU acceleration.

**TinyLlama 1.1B on my 2018 laptop PC (CPU-only), precision: FP16:**

https://github.com/vitoplantamura/OnnxStream/assets/37000260/bb0caee3-3c29-4823-a72b-5be0a4de62ca

**Mistral 7B on Google Colab Free (GPU: Nvidia T4 16GB), precision: FP16:**

https://github.com/vitoplantamura/OnnxStream/assets/37000260/bdca97f1-674c-4dbd-99ac-6ff53a882cf0

### Introduction

OnnxStream can run Stable Diffusion XL Base and Turbo on a Raspberry PI Zero 2. It was designed to run large models on devices with little RAM, thanks to its peculiar architecture where the inference engine is decoupled from the component that provides the weights and which allows the "streaming" of the parameters during inference. The initial idea was to try to run [TinyLlama](https://github.com/jzhang38/TinyLlama) on the Raspberry PI Zero 2, but the performance was in the order of 10 minutes per token, thus configuring no real use case for this solution. So I decided to implement a series of optimizations to run LLMs on desktop computers and servers at acceptable (ie "interactive") speeds with OnnxStream. The biggest difference compared to running on the RPI Zero 2 is the use of the `RamWeightsProvider`, which loads and caches all model parameters into the system's RAM and/or VRAM during the first model inference. All of these optimizations are optional when using the library, so OnnxStream can be used in both scenarios, both on devices with little RAM, and on computers with enough RAM and/or VRAM to load the entire model into memory.

The advantage of the ONNX standard and format is the ability of supporting new models easily: simply export the ONNX file from the original PyTorch model, run [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) on it (optional but recommended) and finally run [onnx2txt](https://github.com/vitoplantamura/OnnxStream/blob/master/onnx2txt/onnx2txt.ipynb) to convert the ONNX file into a format that can be run by OnnxStream. TinyLlama and Mistral7B's example chat application uses OnnxStream models that have been exported directly from the [LlamaForCausalLM](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) and [MistralForCausalLM](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py) classes from the HF Transformers library respectively.

### GPU acceleration

This version of OnnxStream features initial support for GPU acceleration, via cuBLAS for Nvidia cards. Through the "--cuda" parameter it is possible to specify the number of GBs of the model to offload to the GPU. This effectively allows to use all the system memory (RAM+VRAM) and allows a certain level of parallelism between the CPU and the GPU. When partial GPU offloading is enabled, only "static matmuls" are offloaded to the GPU (at least in this version of OnnxStream) for a number of GBs of VRAM that can be specified with the "--cuda" option. Static matmuls are those matmuls where the second operand is static and, in this case, these are the `torch.nn.Linear` layers in PyTorch. The learned weights of all the Linear layers represent almost all of the parameters of a transformer-based LLM and are therefore the most memory bandwidth intensive operations.

### Quantization

Although OnnxStream already supports 8-bit static and dynamic quantization and calibration, only FP16 and FP32 inference is currently supported for LLMs. 8-bit quantization for LLMs will be implemented in the next release (hopefully).

### Code to export TinyLlama to ONNX

<details>
<summary>Click to expand</summary>

```python
import transformers 
import torch
import torch.nn as nn
import onnx

pipeline = transformers.pipeline(
    "text-generation",
    model="PY007/TinyLlama-1.1B-Chat-v0.3",
    torch_dtype=torch.float32,
    device_map="auto",
)

class LlamaModel(nn.Module):
    def __init__(self, model):
        super(LlamaModel, self).__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, position_ids,
                pkv0, pkv1, pkv2, pkv3, pkv4, pkv5, pkv6, pkv7, pkv8, pkv9, pkv10,
                pkv11, pkv12, pkv13, pkv14, pkv15, pkv16, pkv17, pkv18, pkv19, pkv20,
                pkv21, pkv22, pkv23, pkv24, pkv25, pkv26, pkv27, pkv28, pkv29, pkv30,
                pkv31, pkv32, pkv33, pkv34, pkv35, pkv36, pkv37, pkv38, pkv39, pkv40,
                pkv41, pkv42, pkv43):
        past_key_values = (
            (pkv0, pkv1), (pkv2, pkv3), (pkv4, pkv5), (pkv6, pkv7), (pkv8, pkv9), (pkv10,
            pkv11), (pkv12, pkv13), (pkv14, pkv15), (pkv16, pkv17), (pkv18, pkv19), (pkv20,
            pkv21), (pkv22, pkv23), (pkv24, pkv25), (pkv26, pkv27), (pkv28, pkv29), (pkv30,
            pkv31), (pkv32, pkv33), (pkv34, pkv35), (pkv36, pkv37), (pkv38, pkv39), (pkv40,
            pkv41), (pkv42, pkv43))
        o = self.model(use_cache=True, return_dict=True,
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
        pkv = o.past_key_values
        return [o.logits,
               pkv[0][0], pkv[0][1], pkv[1][0], pkv[1][1], pkv[2][0], pkv[2][1], pkv[3][0], pkv[3][1], pkv[4][0], pkv[4][1],
               pkv[5][0], pkv[5][1], pkv[6][0], pkv[6][1], pkv[7][0], pkv[7][1], pkv[8][0], pkv[8][1], pkv[9][0], pkv[9][1],
               pkv[10][0], pkv[10][1], pkv[11][0], pkv[11][1], pkv[12][0], pkv[12][1], pkv[13][0], pkv[13][1], pkv[14][0], pkv[14][1],
               pkv[15][0], pkv[15][1], pkv[16][0], pkv[16][1], pkv[17][0], pkv[17][1], pkv[18][0], pkv[18][1], pkv[19][0], pkv[19][1],
               pkv[20][0], pkv[20][1], pkv[21][0], pkv[21][1] ]

with torch.no_grad():

    dummy_input = (torch.tensor([[1, 2, 3]], dtype=torch.int64),
                  torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.int64),
                  torch.tensor([[3, 4, 5]], dtype=torch.int64))
    for i in range(44):
        dummy_input += (torch.randn(1, 4, 3, 64),)
    input_names = [ "input_ids", "attention_mask", "position_ids",
                "pkv0", "pkv1", "pkv2", "pkv3", "pkv4", "pkv5", "pkv6", "pkv7", "pkv8", "pkv9", "pkv10",
                "pkv11", "pkv12", "pkv13", "pkv14", "pkv15", "pkv16", "pkv17", "pkv18", "pkv19", "pkv20",
                "pkv21", "pkv22", "pkv23", "pkv24", "pkv25", "pkv26", "pkv27", "pkv28", "pkv29", "pkv30",
                "pkv31", "pkv32", "pkv33", "pkv34", "pkv35", "pkv36", "pkv37", "pkv38", "pkv39", "pkv40",
                "pkv41", "pkv42", "pkv43" ]
    output_names = [ "logits",
                    "opkv0", "opkv1", "opkv2", "opkv3", "opkv4", "opkv5", "opkv6", "opkv7", "opkv8", "opkv9", "opkv10",
                    "opkv11", "opkv12", "opkv13", "opkv14", "opkv15", "opkv16", "opkv17", "opkv18", "opkv19", "opkv20",
                    "opkv21", "opkv22", "opkv23", "opkv24", "opkv25", "opkv26", "opkv27", "opkv28", "opkv29", "opkv30",
                    "opkv31", "opkv32", "opkv33", "opkv34", "opkv35", "opkv36", "opkv37", "opkv38", "opkv39", "opkv40",
                    "opkv41", "opkv42", "opkv43" ]
    
    torch.onnx.export(LlamaModel(pipeline.model), dummy_input, "/media/vito/new_disk/Downloads_temp3/test3/TinyLlama.onnx", verbose=False,
        input_names=input_names, output_names=output_names,
        opset_version=14, do_constant_folding=True, export_params=True,
        dynamic_axes={
                'input_ids': {1: 'dim0'},
                'attention_mask': {1: 'dim1'},
                'position_ids': {1: 'dim2'},
                'pkv0': {2: 'dim3'}, 'pkv1': {2: 'dim3'}, 'pkv2': {2: 'dim3'}, 'pkv3': {2: 'dim3'}, 'pkv4': {2: 'dim3'},
                'pkv5': {2: 'dim3'}, 'pkv6': {2: 'dim3'}, 'pkv7': {2: 'dim3'}, 'pkv8': {2: 'dim3'}, 'pkv9': {2: 'dim3'},
                'pkv10': {2: 'dim3'}, 'pkv11': {2: 'dim3'}, 'pkv12': {2: 'dim3'}, 'pkv13': {2: 'dim3'}, 'pkv14': {2: 'dim3'},
                'pkv15': {2: 'dim3'}, 'pkv16': {2: 'dim3'}, 'pkv17': {2: 'dim3'}, 'pkv18': {2: 'dim3'}, 'pkv19': {2: 'dim3'},
                'pkv20': {2: 'dim3'}, 'pkv21': {2: 'dim3'}, 'pkv22': {2: 'dim3'}, 'pkv23': {2: 'dim3'}, 'pkv24': {2: 'dim3'},
                'pkv25': {2: 'dim3'}, 'pkv26': {2: 'dim3'}, 'pkv27': {2: 'dim3'}, 'pkv28': {2: 'dim3'}, 'pkv29': {2: 'dim3'},
                'pkv30': {2: 'dim3'}, 'pkv31': {2: 'dim3'}, 'pkv32': {2: 'dim3'}, 'pkv33': {2: 'dim3'}, 'pkv34': {2: 'dim3'},
                'pkv35': {2: 'dim3'}, 'pkv36': {2: 'dim3'}, 'pkv37': {2: 'dim3'}, 'pkv38': {2: 'dim3'}, 'pkv39': {2: 'dim3'},
                'pkv40': {2: 'dim3'}, 'pkv41': {2: 'dim3'}, 'pkv42': {2: 'dim3'}, 'pkv43': {2: 'dim3'},
        })
```

</details>

### Code to export Mistral 7B to ONNX

<details>
<summary>Click to expand</summary>

```python
import transformers 
import torch
import torch.nn as nn
import onnx
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

class LlamaModel(nn.Module):
    def __init__(self, model):
        super(LlamaModel, self).__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, position_ids,
                pkv0, pkv1, pkv2, pkv3, pkv4, pkv5, pkv6, pkv7, pkv8, pkv9, pkv10,
                pkv11, pkv12, pkv13, pkv14, pkv15, pkv16, pkv17, pkv18, pkv19, pkv20,
                pkv21, pkv22, pkv23, pkv24, pkv25, pkv26, pkv27, pkv28, pkv29, pkv30,
                pkv31, pkv32, pkv33, pkv34, pkv35, pkv36, pkv37, pkv38, pkv39, pkv40,
                pkv41, pkv42, pkv43, pkv44, pkv45, pkv46, pkv47, pkv48, pkv49, pkv50,
                pkv51, pkv52, pkv53, pkv54, pkv55, pkv56, pkv57, pkv58, pkv59, pkv60,
                pkv61, pkv62, pkv63):
        past_key_values = (
            (pkv0, pkv1), (pkv2, pkv3), (pkv4, pkv5), (pkv6, pkv7), (pkv8, pkv9), (pkv10,
            pkv11), (pkv12, pkv13), (pkv14, pkv15), (pkv16, pkv17), (pkv18, pkv19), (pkv20,
            pkv21), (pkv22, pkv23), (pkv24, pkv25), (pkv26, pkv27), (pkv28, pkv29), (pkv30,
            pkv31), (pkv32, pkv33), (pkv34, pkv35), (pkv36, pkv37), (pkv38, pkv39), (pkv40,
            pkv41), (pkv42, pkv43), (pkv44, pkv45), (pkv46, pkv47), (pkv48, pkv49), (pkv50,
            pkv51), (pkv52, pkv53), (pkv54, pkv55), (pkv56, pkv57), (pkv58, pkv59), (pkv60,
            pkv61), (pkv62, pkv63))
        o = self.model(use_cache=True, return_dict=True,
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
        pkv = o.past_key_values
        return [o.logits,
               pkv[0][0], pkv[0][1], pkv[1][0], pkv[1][1], pkv[2][0], pkv[2][1], pkv[3][0], pkv[3][1], pkv[4][0], pkv[4][1],
               pkv[5][0], pkv[5][1], pkv[6][0], pkv[6][1], pkv[7][0], pkv[7][1], pkv[8][0], pkv[8][1], pkv[9][0], pkv[9][1],
               pkv[10][0], pkv[10][1], pkv[11][0], pkv[11][1], pkv[12][0], pkv[12][1], pkv[13][0], pkv[13][1], pkv[14][0], pkv[14][1],
               pkv[15][0], pkv[15][1], pkv[16][0], pkv[16][1], pkv[17][0], pkv[17][1], pkv[18][0], pkv[18][1], pkv[19][0], pkv[19][1],
               pkv[20][0], pkv[20][1], pkv[21][0], pkv[21][1], pkv[22][0], pkv[22][1], pkv[23][0], pkv[23][1], pkv[24][0], pkv[24][1],
               pkv[25][0], pkv[25][1], pkv[26][0], pkv[26][1], pkv[27][0], pkv[27][1], pkv[28][0], pkv[28][1], pkv[29][0], pkv[29][1],
               pkv[30][0], pkv[30][1], pkv[31][0], pkv[31][1] ]

with torch.no_grad():

    dummy_input = (torch.tensor([[1, 2, 3]], dtype=torch.int64),
                  torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.int64),
                  torch.tensor([[3, 4, 5]], dtype=torch.int64))
    for i in range(64):
        dummy_input += (torch.randn(1, 8, 3, 128),)
    input_names = [ "input_ids", "attention_mask", "position_ids",
                "pkv0", "pkv1", "pkv2", "pkv3", "pkv4", "pkv5", "pkv6", "pkv7", "pkv8", "pkv9", "pkv10",
                "pkv11", "pkv12", "pkv13", "pkv14", "pkv15", "pkv16", "pkv17", "pkv18", "pkv19", "pkv20",
                "pkv21", "pkv22", "pkv23", "pkv24", "pkv25", "pkv26", "pkv27", "pkv28", "pkv29", "pkv30",
                "pkv31", "pkv32", "pkv33", "pkv34", "pkv35", "pkv36", "pkv37", "pkv38", "pkv39", "pkv40",
                "pkv41", "pkv42", "pkv43", "pkv44", "pkv45", "pkv46", "pkv47", "pkv48", "pkv49", "pkv50",
                "pkv51", "pkv52", "pkv53", "pkv54", "pkv55", "pkv56", "pkv57", "pkv58", "pkv59", "pkv60",
                "pkv61", "pkv62", "pkv63" ]
                   
    output_names = [ "logits",
                    "opkv0", "opkv1", "opkv2", "opkv3", "opkv4", "opkv5", "opkv6", "opkv7", "opkv8", "opkv9", "opkv10",
                    "opkv11", "opkv12", "opkv13", "opkv14", "opkv15", "opkv16", "opkv17", "opkv18", "opkv19", "opkv20",
                    "opkv21", "opkv22", "opkv23", "opkv24", "opkv25", "opkv26", "opkv27", "opkv28", "opkv29", "opkv30",
                    "opkv31", "opkv32", "opkv33", "opkv34", "opkv35", "opkv36", "opkv37", "opkv38", "opkv39", "opkv40",
                    "opkv41", "opkv42", "opkv43", "opkv44", "opkv45", "opkv46", "opkv47", "opkv48", "opkv49", "opkv50",
                    "opkv51", "opkv52", "opkv53", "opkv54", "opkv55", "opkv56", "opkv57", "opkv58", "opkv59", "opkv60",
                    "opkv61", "opkv62", "opkv63" ]
    
    torch.onnx.export(LlamaModel(model), dummy_input, "/Users/Vito/Desktop/Mistral7BInst/model.onnx", verbose=False,
        input_names=input_names, output_names=output_names,
        opset_version=14, do_constant_folding=True, export_params=True,
        dynamic_axes={
                'input_ids': {1: 'dim0'},
                'attention_mask': {1: 'dim1'},
                'position_ids': {1: 'dim2'},
                'pkv0': {2: 'dim3'}, 'pkv1': {2: 'dim3'}, 'pkv2': {2: 'dim3'}, 'pkv3': {2: 'dim3'}, 'pkv4': {2: 'dim3'},
                'pkv5': {2: 'dim3'}, 'pkv6': {2: 'dim3'}, 'pkv7': {2: 'dim3'}, 'pkv8': {2: 'dim3'}, 'pkv9': {2: 'dim3'},
                'pkv10': {2: 'dim3'}, 'pkv11': {2: 'dim3'}, 'pkv12': {2: 'dim3'}, 'pkv13': {2: 'dim3'}, 'pkv14': {2: 'dim3'},
                'pkv15': {2: 'dim3'}, 'pkv16': {2: 'dim3'}, 'pkv17': {2: 'dim3'}, 'pkv18': {2: 'dim3'}, 'pkv19': {2: 'dim3'},
                'pkv20': {2: 'dim3'}, 'pkv21': {2: 'dim3'}, 'pkv22': {2: 'dim3'}, 'pkv23': {2: 'dim3'}, 'pkv24': {2: 'dim3'},
                'pkv25': {2: 'dim3'}, 'pkv26': {2: 'dim3'}, 'pkv27': {2: 'dim3'}, 'pkv28': {2: 'dim3'}, 'pkv29': {2: 'dim3'},
                'pkv30': {2: 'dim3'}, 'pkv31': {2: 'dim3'}, 'pkv32': {2: 'dim3'}, 'pkv33': {2: 'dim3'}, 'pkv34': {2: 'dim3'},
                'pkv35': {2: 'dim3'}, 'pkv36': {2: 'dim3'}, 'pkv37': {2: 'dim3'}, 'pkv38': {2: 'dim3'}, 'pkv39': {2: 'dim3'},
                'pkv40': {2: 'dim3'}, 'pkv41': {2: 'dim3'}, 'pkv42': {2: 'dim3'}, 'pkv43': {2: 'dim3'}, 'pkv44': {2: 'dim3'},
                'pkv45': {2: 'dim3'}, 'pkv46': {2: 'dim3'}, 'pkv47': {2: 'dim3'}, 'pkv48': {2: 'dim3'}, 'pkv49': {2: 'dim3'},
                'pkv50': {2: 'dim3'}, 'pkv51': {2: 'dim3'}, 'pkv52': {2: 'dim3'}, 'pkv53': {2: 'dim3'}, 'pkv54': {2: 'dim3'},
                'pkv55': {2: 'dim3'}, 'pkv56': {2: 'dim3'}, 'pkv57': {2: 'dim3'}, 'pkv58': {2: 'dim3'}, 'pkv59': {2: 'dim3'},
                'pkv60': {2: 'dim3'}, 'pkv61': {2: 'dim3'}, 'pkv62': {2: 'dim3'}, 'pkv63': {2: 'dim3'},
        })
```

</details>

### Code to export the vocabulary of TinyLlama and Mistral 7B

<details>
<summary>Click to expand</summary>

```python
from sentencepiece import SentencePieceProcessor

model_path = "/Users/Vito/Downloads/tokenizer.model"

sp=SentencePieceProcessor(model_file=model_path)

c = ""
for i in range(sp.vocab_size()):
 s = sp.get_score(i)
 si = int(s)
 assert float(si) == s
 c += str(si) + "," + str(sp.id_to_piece(i).replace('▁', ' ').encode('utf-8'))[2:-1] + "\n"

c = c[:-1]

with open('/Users/Vito/Downloads/vocab.txt', 'w') as f:
 f.write(c)
```

</details>
