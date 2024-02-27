from transformers import LlamaConfig,LlamaTokenizer
from myllama import LlamaForCausalLM
from accelerate import init_empty_weights,infer_auto_device_map
import torch

cuda_list = '0,1,2,3,4,5,6,7'.split(',')
memory = '44GiB'
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
sents=['你是谁']
ids = tokenizer(sents,max_length=1800,padding=True,truncation=True,return_tensors="pt")
ids = ids.to(model.device) 
outputs = model.generate(**ids, do_sample=False)
text=tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)