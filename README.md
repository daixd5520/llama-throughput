
# 7b正常推理-有输出，交互
注：只能进行batchsize=1的测试。此仓库中的example_chat_completion.py和generation.py已做更改，加入了计时器和throughput输出
注2:这种运行方式无法读取hf格式（.bin/.safetensors）的模型，需要非hf格式的
## 环境、模型准备
1. `conda create -n llama python=3.11`
2. `pip install -r requirements.txt`
3. 下载hf格式的llama2-7b(或llama2-7b-chat)
## 运行方式
1. 
```shell
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir /path/to/Llama2-7b-chat \
    --tokenizer_path /path/to/Llama2-7b-chat/tokenizer.model \
    --max_seq_len 4096 --max_batch_size 1
```

2. 随便输入一个prompt，忽略输出的时间和throuhgput（第一个问题随便问，不要记录，因为first-token-time很大，为0.5s左右；随后的问题可以直接问，不用clear掉历史记录，不用重新加载模型）

3. 输入一定长度的prompt（32，64，。。。，等），记录输出的几个指标。

# 批次推理-无输出打印，无交互
## 运行方式
1. 
```shell
torchrun --nproc_per_node 1 ori_example_chat_completion.py \
    --ckpt_dir /root/llama2-7b-chat/ \
    --tokenizer_path /root/llama2-7b-chat/tokenizer.model \
    --max_seq_len 3500 --max_batch_size 8 --input_len 2048
```
input_len:想要输入的seq长度，根据这个长度，代码会自动生成这个长度的prompt

2. 真实的batch throughput需要进行一个计算：输出中的(All generated tokens)*batchsize大小/(输出中的总耗时)

比如：batchsize为8的输出如下
```
>>>>>>>>>>>>First token 耗时:  2.375967502593994  s                        
                                                                           
--------------Throughput: 41                                               
                                                                           
--------------All generated tokens: 105.0                                  
                                                                           
--------------ms/token: 24.20133181980678                                  
总耗时：13.4586021900177 s  
```
则真实throughput就是105*8/13.4586021900177=62.4136138463942

# 巨大模型推理-pp
（以LLaMA2-70B 多卡推理为例，其他模型的切分自行修改chat_demo_tp8.py）
切分方式：按层切分（所以严格来说是pp）

本项目为了在模型实现代码内加计时器和throughput记录，做了：
- 将transformer库里的modeling_llama.py拷贝到本项目的myllama.py。
- 并且在chat_demo_tp8.py中 将LlamaForCausalLM从myllama引入，而不是transformer库里引入

注：chat_demo_tp8.py中，memory = '20GiB'的设置：控制每个卡上最多load模型的多大的部分；若设置为44GiB，那么会将第一张卡放满44GiB，再放下面的卡

70B的模型大概占用136GiB显存，平均分到八张卡上差不多20GiB/p

## 运行方式
1. 修改chat_demo_tp8.py中的line 15 模型目录
2. `python chat_demo_tp8.py`

3. 随便输入一个prompt，忽略输出的时间和throuhgput（第一个问题随便问，不要记录，因为first-token-time很大，为0.5s左右；随后的问题可以直接问，不用clear掉历史记录，不用重新加载模型）

4. 输入一定长度的prompt（32，64，。。。，等），记录输出的几个指标。