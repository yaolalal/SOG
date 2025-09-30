import os
import torch
from transformers import TrainingArguments,DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

from reference.llm_loader import reload_model_and_tokenizer

from datasets import Dataset,load_dataset,concatenate_datasets
from transformers import AutoTokenizer
import re
import math
import json
from reference.utils import print_dict,get_tokenizer
from tqdm import tqdm
import logging
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

model_name = "llama3.2-3b"
model_path = f"/data1/wujingyao-20354/wujingyao/codes/models/llama-3.2-3b-instruct"
tokenizer_path = model_path
output_dir = f"./models/structure-pretrain-{model_name}-2stage"
device_map = "auto"

# 😭 如果不幸中断 请重写reload_path: short/medium/long的最后一个ckpt
# reload_path = ''
# model_path = reload_path
# tokenizer_path = reload_path

per_device_train_batch_size_dict = {
    "short_texts_path": 128,
    "medium_texts_comp": 32,
    "long_texts_match": 1,
}
gradient_accumulation_steps_dict = {
    "short_texts_path": 1,
    "medium_texts_comp": 4,
    # "long_texts_match": 16,
}
num_train_epochs_dict = {
    "short_texts_path": 1,
    "medium_texts_comp": 3,
    "long_texts_match": 3,
}

class FixedDeviceSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取模型实际所在的 device（例如 cuda:3）
        device = next(model.parameters()).device

        # 将所有 Tensor 类型的 inputs 搬到对应 device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        num_items_in_batch = num_items_in_batch.item()

        return super().compute_loss(model, inputs, return_outputs,num_items_in_batch)

def wrap_struct_token(token_id):
    output = f"""
    [Expected output]
    Structure token: <struct><gstruct_{token_id}></struct>
    """
    return output


model, tokenizer = reload_model_and_tokenizer(tokenizer_path=tokenizer_path, model_path=model_path, device_map=device_map, lora=True)
embedding_size = model.get_input_embeddings().weight.shape
print(f"Embedding layer size after resize: {embedding_size}")
lm_head_size = model.lm_head.weight.shape
print(f"LM head size after resize: {lm_head_size}")
tokenizer.pad_token = tokenizer.eos_token
print('tokenizer config:',tokenizer.pad_token,tokenizer.padding_side,len(tokenizer.get_vocab()))
model = prepare_model_for_kbit_training(model)

# Unfreeze and mask gradients for new token embeddings
# model.model.embed_tokens.weight.requires_grad = True
# model.lm_head.weight.requires_grad = True

# Define the start index for new tokens
num_added_tokens = 256
vocab_size = len(tokenizer.get_vocab())
new_token_start_id = vocab_size - num_added_tokens

# Gradient masking to only train new tokens
def mask_grad_embed(grad):
    mask = torch.zeros_like(grad)
    mask[new_token_start_id:] = 1
    return grad * mask

def mask_grad_lmhead(grad):
    mask = torch.zeros_like(grad)
    mask[new_token_start_id:] = 1
    return grad * mask

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        # modules_to_save=["lm_head","embed_tokens"],
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    )

tool,assistant_name = get_tokenizer(tokenizer_path=tokenizer_path)
print("🌟 处理 **结构 tokens 全局距离** 语料：corpus/cora/struct_code_train_corpus.txt")
# 示例：读取 txt 文件，每行为一个训练样本
with open("./corpus/struct_code_train_corpus.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print('Length(raw):',len(lines))
dataset_path = Dataset.from_dict({"text": lines})
print('Example:')
print(dataset_path[0]['text'])

print("\n🌟 处理 **结构 tokens 相对距离** 语料：corpus/cora/struct_code_similarity_corpus.jsonl")
dataset_comp = load_dataset("json", data_files="./corpus/struct_token_similarity.jsonl",split='train')
print('Length(raw):',len(dataset_comp))
def process_io(row):
    messages = [
      {"role": "user", "content": row['instruction']},
      {"role": assistant_name, "content": row['output']},
    ]
    row["text"] = tool.apply_chat_template(messages, tokenize=False)+"<|end_of_text|>"
    return row  
dataset_comp = dataset_comp.map(process_io)
dataset_comp = dataset_comp.remove_columns(['instruction','output'])
print('Example:')
print(dataset_comp[0]['text'])


# print("\n🌟 处理 **结构 tokens 子图匹配** 语料：corpus/cora/ego_graph_descriptions_corpus.json")
# file_path = "./corpus/graph_desc_token_pairs.json"
# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
# print('Length(raw) of train/validation/test:',len(data['train']),len(data['valid']),len(data['test']))
# prefix_info = """
# Given the structural description below, identify the best matching structure token for the graph.
# This description outlines the graph structure starting with a center node which has largest degree, illustrating the connections between different nodes.
# """
# for split,data_tmp_list in data.items():
#     pbar = tqdm(data_tmp_list, desc="Processing dataset_match")
#     data_res_list = []
#     for g in pbar:
#         code = g['code']
#         desc = g['desc']
#         output = wrap_struct_token(code)
#         messages = [
#             {"role": "user", "content": prefix_info},
#             {"role": "user", "content": desc},
#             {"role": assistant_name, "content": output},
#         ]
#         text = tool.apply_chat_template(messages, tokenize=False)+"<|end_of_text|>"
#         data_res_list.append({"text": text})
# dataset_match = Dataset.from_list(data_res_list)
# print('Wrapping to dataset:',dataset_match)
# print('Example:')
# print(dataset_match[0]['text'])


datasets = {
    # "short_texts_path": dataset_path,
    "medium_texts_comp": dataset_comp,
    # "long_texts_match": dataset_match,       
}

# 😭如果不幸中断，请重写 datasets：跳过已经训练过的datasets
# datasets = {
#     "medium_texts_comp": dataset_comp,
# }

# prev_checkpoint = None

for stage_id, (name, dataset) in enumerate(datasets.items()):
    print(f"🔄 Stage {stage_id+1}: Training on {name}")
    per_device_train_batch_size = per_device_train_batch_size_dict[name]
    gradient_accumulation_steps = gradient_accumulation_steps_dict[name]
    num_train_epochs = num_train_epochs_dict[name]
    
    # 配置训练参数
    max_seq_length = max([len(tokenizer(example["text"])['input_ids']) for example in dataset])
    print(f"Max sequence length for {name}: {max_seq_length}")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 表示这是 causal LM，而不是 masked LM
    )
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        group_by_length=True,  # ✅ 关键参数，按长度分组(roughly)
        log_level="debug",
        save_strategy="no",
        # logging_steps=100, # 第一次是100，这太少了，cora数据集总共也就500步，改成10
        logging_steps=10,
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        # resume_from_checkpoint=prev_checkpoint,
        # overwrite_output_dir=True,  # ✅ 覆盖输出目录
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     max_seq_length=max_seq_length,
    #     tokenizer=tokenizer,
    #     args=training_arguments,
    #     data_collator=data_collator,  # ✅ 动态 padding
    # )
    trainer = FixedDeviceSFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,  # ✅ 动态 padding
    )
    print("TrainingArguments:")
    for key, value in vars(training_arguments).items():
        print(f"  {key}: {value}")
    
    trainer.model.base_model.model.model.embed_tokens.weight.requires_grad = True
    trainer.model.base_model.model.lm_head.weight.requires_grad = True

    trainer.model.base_model.model.model.embed_tokens.weight.register_hook(mask_grad_embed)
    trainer.model.base_model.model.lm_head.weight.register_hook(mask_grad_lmhead)
    print('[INFO] judge: trainer.model.base_model.model vs model',trainer.model.base_model.model == model)

    # 打印模型可训练参数
    trainer.model.print_trainable_parameters()
    # 检查 embedding 层
    embed_grad_status = trainer.model.base_model.model.model.embed_tokens.weight.requires_grad
    print(f"Embedding layer requires_grad: {embed_grad_status}")
    # 检查 lm_head 层
    lm_head_grad_status = trainer.model.base_model.model.lm_head.weight.requires_grad
    print(f"LM head requires_grad: {lm_head_grad_status}")
    print("Trainable parameters:")
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(name)

    trainer.train()
    trainer.save_model(output_dir)


print("🌟 Training completed for all stages!")
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



