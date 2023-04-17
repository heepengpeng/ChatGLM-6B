import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

model_name = "chatglm-6b" # 模型名 或 模型路径
checkpoint_path = ".\output\checkpoint-500" # 模型checkpoint路径
pre_seq_len = 128 # 模型前缀长度 跟你训练的PRE_SEQ_LEN一致

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.pre_seq_len = pre_seq_len
model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)

prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model.half().cuda()

print(model.chat(tokenizer, "安装MindSpore版本: GPU、CUDA 10.1、0.5.0-beta"))