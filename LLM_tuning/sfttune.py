import os
import pickle
from datasets import Dataset, concatenate_datasets
import multiprocessing
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch,transformers
from reference.llm_loader import reload_model_and_tokenizer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
from reference.utils import get_tokenizer
import random
import argparse
random.seed(42)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class FixedDeviceSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取模型实际所在的 device（例如 cuda:3）
        device = next(model.parameters()).device

        # 将所有 Tensor 类型的 inputs 搬到对应 device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        num_items_in_batch = num_items_in_batch.item()

        return super().compute_loss(model, inputs, return_outputs,num_items_in_batch)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="corpus/old/ds_dict_ori.pkl")
parser.add_argument('--data_prefix', type=str, default="ori")
parser.add_argument('--per_device_train_batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--united_list', type=str, default='', help='Comma-separated list of items to process（others ignore）')

args = parser.parse_args()
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
num_train_epochs =args.num_train_epochs
data_path = args.data_path
data_prefix = args.data_prefix
united_list = args.united_list.split(',') if args.united_list else []
united = "-".join(united_list)
print(args)
###########################################################################################################################
low_atten = False
########### 使用 7b 模型##################
model_name = "llama-2-7b"
reload_path = "/home/wujingyao/codes/experiment/GraphLLM-graph/models/second-stage-llama-2-7b/long_texts_match/checkpoint"
########### 使用 3b 模型##################
# model_name = "llama3.2-3b"
# reload_path = "/home/wujingyao/codes/experiment/GraphLLM-graph/models/second-stage-llama3.2-3b/checkpoints/long_texts_match/checkpoint-1011"
output_dir = f"./fastoutput/{model_name}-{united}-{data_prefix}" # 使用原始数据（没有均衡过）+ 不调整embedding层（因为没有新增tokens）
with open(data_path, "rb") as f:
    ds_dict = pickle.load(f)

# 🌟🌟🌟 加载模型 🌟🌟🌟
model,tokenizer = reload_model_and_tokenizer(tokenizer_path=reload_path,model_path=reload_path,device_map="auto",lora=True,low_attention=low_atten)
# model,tokenizer = reload_model_and_tokenizer(tokenizer_path=tokenizer_path,model_path=model_path,device_map=device_map,lora=True)
print("tokenizer.vocab size:",len(tokenizer.get_vocab()))
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("pad_token is None, set to eos_token")
model.config.pad_token_id = tokenizer.pad_token_id

assistant_name = "LLM Assistant"
system_prompt = "You are a helpful and reliable assistant."
format_system_prompt = "<|start_header_id|>system<|end_header_id|>\n\n"+system_prompt+"<|eot_id|>"
tokenizer.chat_template = "<|begin_of_text|>"+format_system_prompt+"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>"+assistant_name+"<|end_header_id|>\n\n' }}{% endif %}"
num_added_tokens = 256
vocab_size = len(tokenizer.get_vocab())
new_token_start_id = vocab_size - num_added_tokens
# 打印新添加的 token id 和对应的 tokens
for i in range(num_added_tokens):
    if i in [0,1,128,255]:
        print(f"Token ID {new_token_start_id + i}: {tokenizer.convert_ids_to_tokens(new_token_start_id + i)}")

# Freeze all parameters
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    log_level="debug",
    save_strategy="epoch",
    # logging_steps=100, # 第一次是100，这太少了，总共也就500步，改成10
    logging_steps=10,
    learning_rate=1e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    overwrite_output_dir = False
)

if united_list:
    train_datasets = [ds for ds_name,ds in ds_dict['train'].items() if ds_name in united_list]
    valid_datasets = [ds for ds_name,ds in ds_dict['valid'].items() if ds_name in united_list]
else:
    train_datasets = list(ds_dict['train'].values())
    valid_datasets = list(ds_dict['valid'].values())
ds_train = concatenate_datasets(train_datasets)
ds_valid = concatenate_datasets(valid_datasets)
max_length_train = max([len(tokenizer(example["text"])['input_ids']) for example in ds_train])
max_length_valid = max([len(tokenizer(example["text"])['input_ids']) for example in ds_valid])
max_length = max(max_length_train, max_length_valid)
# train_dataset = Dataset.from_list([
#     preprocess_function(ex, tokenizer, max_length=max_length_train)
#     for ex in ds_train
# ])
# valid_dataset = Dataset.from_list([
#     preprocess_function(ex, tokenizer, max_length=max_length_valid)
#     for ex in ds_valid
# ])

trainer = FixedDeviceSFTTrainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_length,
        tokenizer=tokenizer,
        args=training_arguments,
)
# trainer = FixedDeviceSFTTrainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         peft_config=peft_config,
#         tokenizer=tokenizer,
#         args=training_arguments,
#     )
trainer.train()

# 打印模型可训练参数
trainer.model.print_trainable_parameters()
for name, param in trainer.model.named_parameters():
    if param.requires_grad:
        print(name)

# 检查 embedding 层
embed_grad_status = model.model.embed_tokens.weight.requires_grad
print(f"Embedding layer requires_grad: {embed_grad_status}")
# 检查 lm_head 层
lm_head_grad_status = model.lm_head.weight.requires_grad
print(f"LM head requires_grad: {lm_head_grad_status}")

