import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import time

from transformers import LlamaConfig,LlamaTokenizer
from myllama import LlamaForCausalLM
from accelerate import init_empty_weights,infer_auto_device_map
import torch

cuda_list = '0,1,2,3,4,5,6,7'.split(',')
memory = '20GiB'
model_path = '/root/Llama2-70b-chat-hf'
no_split_module_classes = LlamaForCausalLM._no_split_modules

max_memory = {int(cuda):memory for cuda in cuda_list}
config = LlamaConfig.from_pretrained(model_path)
with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) #torch_dtype=torch.float16这个很重要

device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) #自动划分每个层的设备
model = LlamaForCausalLM.from_pretrained(model_path,device_map=device_map, torch_dtype=torch.float16)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 LLaMA2-70B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用 LLaMA2-70B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 LLaMA2-70B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        sents=[query]
        ids = tokenizer(sents,max_length=1800,padding=True,truncation=True,return_tensors="pt")
        ids = ids.to(model.device) 
        #######################改：增start#########################
        model.model.fwd_num = 0
        model.model.seq_len = 0
        model.model.encode_time = 0
        model.model.decode_time = 0
        total_response = ""
        #######################改：增end###########################
        outputs = model.generate(**ids, do_sample=False)
        text=tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(text)

        print("Tokens generated in 1 sec（第一秒生成token数）: %d" % (model.model.one_sec_tokens))
        print("First token time（生成第一个token耗时）: %.4f ms" % (model.model.first_token_time))
        print("Generated token count（总生成token数）:%d" % (model.model.fwd_num - 1))
        print("Time per token（平均生成每个token用时）:%.4f ms" % ((model.model.encode_time+model.model.decode_time)/(model.model.fwd_num - 1)))
        model.model.one_sec_tokens=0


if __name__ == "__main__":
    main()