# 正常推理
测试说明
注意
此项目中的example_chat_completion.py已更新可以进行交互式测试

第一个问题随便问，不要记录，因为first-token-time很大，为0.5s左右；随后的问题可以直接问，不用clear掉历史记录，不用重新加载大模型
以前的代码first token time大，是因为非交互式生成只输出第一次生成时的first-token-time（0.5s左右）
run.sh中的max_batch_size改小的原因是，max_batch_size=6，4，2时会Cuda OOM

此仓库中的example_chat_completion.py和generation.py已做更改，加入计时器和throughput输出；

bash run.sh

# 巨大模型推理
（以LLaMA2-70B 多卡推理为例，其他模型的切分自行修改chat_demo_tp8.py）
切分方式：按层切分（所以严格来说是pp）
为了在模型实现代码内加计时器和throughput记录，我：
- 将transformer库里的modeling_llama.py拷贝到本项目的myllama.py。
- 并且在chat_demo_tp8.py中 将LlamaForCausalLM从myllama引进

chat_demo_tp8.py中，memory = '20GiB'的设置：控制每个卡上最多load模型的多大的部分；若设置为44GiB，那么会将第一张卡放满44GiB，再放下面的卡

70B的模型大概占用136GiB显存，平均分到八张卡上差不多20GiB/p

## 运行方式
python chat_demo_tp8.py

交互式对话，无历史记录
