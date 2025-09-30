# 加载数据（json文件）
import json
import numpy as np
import os
import re
import torch,multiprocessing
from datasets import DatasetDict,Dataset
from transformers import AutoTokenizer


dataset_name = 'cora'
data_path = '/home/wujingyao/codes/experiment/Quantization/VQGraph-my/cora.json'
with open(data_path, 'r') as f:
    data = json.load(f)
print("🌟Loaded {} nodes. Data format:".format(len(data)))
print(data[0])


tokenizer_path = "/data1/wujingyao-20354/wujingyao/codes/models/llama-3.2-3b-instruct-graph/update_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
assistant_name = "LLM Assistant"
system_prompt = "You are a helpful and reliable assistant."
format_system_prompt = "<|start_header_id|>system<|end_header_id|>\n\n"+system_prompt+"<|eot_id|>"
tokenizer.chat_template = "<|begin_of_text|>"+format_system_prompt+"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>"+assistant_name+"<|end_header_id|>\n\n' }}{% endif %}"


prompt = """Classify the node into one of above categories:
Text attribute: {text_attribute}
Ego Graph: <struct>{structure_token}</struct>
Neighbors attributes:
{one_hop_str}{two_hop_str}
"""

def extract_title_and_abstract(data_list):
    """
    从包含 'raw_text' 字段的字典列表中提取标题和摘要。

    'raw_text' 字段的期望格式为:
    'Title: <您的标题内容>  \tAbstract: <您的摘要内容>'

    参数:
    data_list (list): 一个字典列表。每个字典预计包含一个 'raw_text' 键。

    返回:
    list: 一个字典列表，每个字典包含成功提取的 'title' 和 'abstract'。
          如果某个条目无法提取，则会跳过该条目。
    """

    # 正则表达式模式解释:
    # - r"Title:": 匹配字符串 "Title:"。
    # - (?P<title>.*?)":
    #   - (?P<title>...) : 捕获匹配的内容到名为 'title' 的组中。
    #   - .*? : 匹配任意字符 ('.') 零次或多次 ('*')，但尽可能少地匹配 ('?') (非贪婪模式)。
    #           这确保它在遇到 "\s*\tAbstract:" 之前停止。
    # - \s*: 匹配零个或多个空白字符 (例如空格、制表符本身之外的空白)。
    # - \t: 匹配一个制表符。
    # - Abstract:: 匹配字符串 "Abstract:"。
    # - (?P<abstract>.*)":
    #   - (?P<abstract>...) : 捕获匹配的内容到名为 'abstract' 的组中。
    #   - .* : 匹配任意字符 ('.') 零次或多次 ('*') (贪婪模式，会匹配到字符串末尾)。
    # - re.DOTALL: 使 '.' 特殊字符可以匹配换行符，这样标题或摘要可以跨越多行。
    pattern = re.compile(r"Title:(?P<title>.*?)\s*\tAbstract:(?P<abstract>.*)", re.DOTALL)
    not_found = 0
    not_found_list = []
    not_found_centers = []
    for item in data_list:
        if not isinstance(item, dict):
            # 您可以选择记录或处理非字典类型的条目
            print(f"警告: 跳过非字典条目: {item}")
            continue

        raw_text = item.get('raw_text')

        if not raw_text or not isinstance(raw_text, str):
            # 'raw_text' 字段缺失或不是字符串，跳过
            print(f"警告: 跳过 'raw_text' 缺失或无效的条目: {item}")
            continue

        match = pattern.search(raw_text)
        if match:
            # .strip() 用于去除捕获内容两端的空白字符 (包括换行符和多余的空格)
            title = match.group("title").strip()
            abstract = match.group("abstract").strip()
            item['title'] = title
            item['abstract'] = abstract
        else:
            # 如果需要，可以记录那些 'raw_text' 格式不匹配的条目
            # print(f"信息: 未在以下 'raw_text' 中匹配到模式: {raw_text[:100]}...")
            not_found += 1
            not_found_list.append((raw_text, item['node']))
            not_found_centers.append(item['node'])
            stripped_line = raw_text.lstrip()
            if stripped_line.startswith("Title:"):
                item['title'] = stripped_line.split("Title:", 1)[1].strip()
                item['abstract'] = None
            elif stripped_line.startswith("Abstract:"):
                item['abstract'] = stripped_line.split("Abstract:", 1)[1].strip()
                item['title'] = None
            else:
                item['title'] = None
                item['abstract'] = None           
    print(f"未匹配到模式的条目数: {not_found}")

    only_title = []
    only_abstract = []
    bad = []
    for (raw_text, node) in not_found_list:
        stripped_line = raw_text.lstrip()
        if stripped_line.startswith("Title:"):
            only_title.append(node)
        elif stripped_line.startswith("Abstract:"):
            only_abstract.append(node)
        else:
            bad.append(node)
    print(f"Only titles: {only_title}")
    print(f"Only abstracts: {only_abstract}")
    print(f"Bad: {bad}")

    return not_found_list,not_found_centers


def process(row,verbose=False):
   # 1.文本属性
   if row['title'] is not None or row['abstract'] is not None:
      text_attribute = row['raw_text']
   else:
      text_attribute = ''

   # 2.结构信息
   structure_token = f'<struct_{row["structure_code"]}>'

   # 3.邻居属性
   one_hop_str = ''
   for i in range(len(row['one_hop_positions'])):
      neigh = row["one_hop_neighbors"][i]
      pos = row['one_hop_positions'][i]
      label = data[neigh]['label_name']
      if data[neigh]['title'] is not None:
         neigh_text = data[neigh]['title']
      elif data[neigh]['abstract'] is not None:
         neigh_text = data[neigh]['abstract'][:66]+'...'
      else:
         neigh_text = ''
         if verbose:
            print(f'Bad node appear in one_hop_neighbors rank {pos}:',row['node'])
      one_hop_str += f'One hop rank {pos} neigh attribute:{neigh_text}, label:{label},\n'
      
   two_hop_str = ''
   for i in range(len(row['two_hop_positions'])):
      neigh = row["two_hop_neighbors"][i]
      pos = row['two_hop_positions'][i]
      label = data[neigh]['label_name']
      if data[neigh]['title'] is not None:
         neigh_text = data[neigh]['title']
      elif data[neigh]['abstract'] is not None:
         neigh_text = data[neigh]['abstract'][:66]+'...'
      else:
         neigh_text = ''
         if verbose:
            print(f'Bad node appear in two_hop_neighbors rank {pos}:',row['node'])
      two_hop_str += f'Two hop rank {pos} neigh attribute:{neigh_text}, label:{label},\n'
   
   # 4.构建提示
   whole_text = prompt.format(text_attribute=text_attribute,
                              structure_token=structure_token,
                              one_hop_str=one_hop_str,
                              two_hop_str=two_hop_str)
   row['instruction'] = whole_text
   row['response'] = row['label_name']
          
   messages = [
      {"role": "user", "content": row['instruction']},
      {"role": assistant_name, "content": row['response']},
   ]
   row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)+"<|end_of_text|>"
   return row

def train_val_test_split(dataset_name):
    f_path = f'/home/wujingyao/codes/ experiment/Quantization/VQGraph-my/new_node_feats/{dataset_name}/splits.json'
    with open(f_path,'r') as f:
        splits = json.load(f)
    train_indices = splits['train']
    val_indices = splits['val']
    test_indices = splits['test']
    return train_indices,val_indices,test_indices


def load_data(verbose=True):
    _,_ = extract_title_and_abstract(data)  
    dataset = Dataset.from_list(data)
    ds = dataset.map(
        process,
        num_proc= multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    train_indices,val_indices,test_indices = train_val_test_split(dataset_name)
    train_dataset = ds.select(train_indices)
    val_dataset = ds.select(val_indices)
    test_dataset = ds.select(test_indices)

    ds_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    if verbose:
        print('🌟Final:',ds_dict)
        print('🌟Format:')
        print(ds_dict['train'][0]['text'])

    return ds_dict


def get_token_probability(model, input_tokens, target_token):
    with torch.no_grad():
        outputs = model(input_tokens)
    # get the logits for our model output
    logits = outputs.logits[:, -1, :]
    # calculate the softmax probabilities
    probs = torch.softmax(logits, dim=-1)
    token_prob = probs[0, target_token]
    return token_prob

def print_probabilibites_on_test(model, tokenizer, print_tokens):
    question = "Continue writing. The most important information of a graph is ." 
    question = "Ego Graph: <struct>"
    messages = [{"role": "user", "content": question}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True)
    tokens = tokens.to(model.device)

    for token in print_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        prob = get_token_probability(model, tokens, token_id)
        print(f"Token: {token}, Probability: {prob:.6f}")

def print_dict(dictionary):
    for key, value in dictionary.items():
        if type(value) == str:
            print(key,':\n', value)
        else:
            print(key,' : ', value)
def wrap_struct_token(example):
    # 匹配形如 <gstruct_180> 的结构并替换
    example_new = re.sub(r'(<struct_\d+>)', r'<struct>\1</struct>', example)
    return example_new

def get_tokenizer(tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = "/data1/wujingyao-20354/wujingyao/codes/models/llama-3.2-3b-instruct-graph/update_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    assistant_name = "LLM Assistant"
    system_prompt = "You are a helpful and reliable assistant."
    format_system_prompt = "<|start_header_id|>system<|end_header_id|>\n\n"+system_prompt+"<|eot_id|>"
    tokenizer.chat_template = "<|begin_of_text|>"+format_system_prompt+"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>"+assistant_name+"<|end_header_id|>\n\n' }}{% endif %}"
    return tokenizer,assistant_name


if __name__ == '__main__':
    load_data()

    

