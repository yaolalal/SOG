import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import statistics
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import random
import transformers
from tqdm import tqdm
import gc
import logging
import os
import argparse
import copy

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# For CUDA version issues
low_atten = False
random.seed(42)  # 设置随机种子以确保结果可复现

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--data_path', type=str, default='corpus/ds_dict.pkl', help='Path to the test dataset')
parser.add_argument('--times', type=int, default=3, help='Number of times to evaluate')
parser.add_argument('--log_file', type=str, default='eval.log', help='Log file path')

args = parser.parse_args()
batch_size = args.batch_size
data_path = args.data_path
log_file = args.log_file
times = args.times
# ignore_list is designed to ignore some tasks that are not for evaluation
# attention_list is designed to evaluate some tasks that are targeted for evaluation
ignore_list = [] # eg. ignore_list = ['BBBP_p_np','BACE_Class']
attention_list = [] # eg. attention_list = ['BBBP_p_np','BACE_Class']

def setup_logger(log_file='eval.log', log_level=logging.INFO):
    logger = logging.getLogger("eval_logger")
    logger.setLevel(log_level)

    # 防止重复添加 handler
    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # 文件输出
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)

        # 格式
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加 handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger(log_file=log_file)

base_dir = "./fastoutput" # 存储模型的根目录
prefix = "llama-2-7b-"
suffixes = ("")   
llama_dirs = [
    name for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name)) and name.startswith(prefix) and name.endswith(suffixes)
]

with open(data_path, "rb") as f:
    ds_dict = pickle.load(f)
logger.info(f"Loaded dataset from {data_path}.")
if 'test' not in ds_dict:
    ds_dict = {'test': ds_dict}
    logger.info(f"Loaded dataset from {data_path} does not contain 'test' key. Wrapping the dataset into a dictionary with 'test' key.")
test_datasets = ds_dict['test']

stripped_names = [] 
for name in llama_dirs:
    selected_dataset = None
    max_len = -1
    for dataset in test_datasets.keys():
        if dataset in name and len(dataset) > max_len:
            selected_dataset = dataset
            max_len = len(dataset)
    stripped_names.append(copy.deepcopy(selected_dataset))

checkpoint_paths = [
    os.path.join(base_dir, name)
    for name in llama_dirs
    if os.path.isdir(os.path.join(base_dir, name))
]
try:
    model = AutoModelForCausalLM.from_pretrained(checkpoint_paths[0],device_map="auto",low_cpu_mem_usage=True)
    del model
    torch.cuda.empty_cache()
    gc.collect()    
except Exception as e:
    print(f"[ERROR] Failed to load from {checkpoint_paths}: {e}")
    # 遍历 checkpoint_paths 中的每个路径下的所有以checkpoint开头的文件夹，并构成新的路径列表
    new_checkpoint_paths = []
    for path in checkpoint_paths:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("checkpoint")]
        for subdir in subdirs:
            new_checkpoint_paths.append(os.path.join(path, subdir))
    checkpoint_paths = new_checkpoint_paths
    stripped_names = []
    # 获取对应的 stripped_names
    for name in checkpoint_paths:
        father_name = os.path.basename(os.path.dirname(name))
        stripped_names += [father_name[len(prefix):-len(suffixes)]]

# 🌟🌟🌟 遍历所有模型和相应的任务 🌟🌟🌟
for k,checkpoint_path in zip(stripped_names,checkpoint_paths):
    if k in ignore_list:
        logger.info(f"skipping {k}...")
        continue
    if attention_list and k not in attention_list:
        logger.info(f"skipping {k}...")
        continue
    logger.info("Evaluating for {}:".format(checkpoint_path))
    # 🌟🌟🌟 加载模型 🌟🌟🌟
    if low_atten:
        compute_dtype = torch.bfloat16
        attn_implementation = 'sdpa'
    elif torch.cuda.is_bf16_supported():
        # os.system('pip install flash_attn')
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    print(compute_dtype, attn_implementation)
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path,device_map="auto",torch_dtype=compute_dtype,attn_implementation=attn_implementation,low_cpu_mem_usage=True)
        print(f"[SUCCESS] Loaded from: {checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load from {checkpoint_path}: {e}")
        continue     
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def load_model():
        # tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        return pipeline

    # 🌟🌟🌟 🤖评测模型分类性能🥇 🌟🌟🌟
    all_results = {}
    td = test_datasets[k]
    max_length= max([len(tokenizer(example["instruction"]+example['response'])['input_ids']) for example in td])
    print('max_length:', max_length)
    max_response_length= max([len(tokenizer(example['response'])['input_ids']) for example in td])
    print('max_response_length:', max_response_length)

    # test_dataset = td.map(test_process, batched=True)
    test_dataset = td
    print(test_dataset)

    accuracy_list = []
    logger.info('acc1 is calculated by match_count/total; acc2 is calculated by sklearn')
    for i in range(times):
        print(f"Experiment {i+1}:")
        messages = [[{"role": "user", "content": user_prompt}] for user_prompt in test_dataset['instruction']]
        pipeline = load_model()
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        results = []

        for i in tqdm(range(0,len(prompt),batch_size)):
            batch_prompt = prompt[i:i+batch_size]
            outputs = pipeline(
                batch_prompt,
                max_new_tokens=100,
                batch_size=batch_size,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            results += [out[0]["generated_text"] for out in outputs]

        # 计算准确率 -- 匹配真实结果是否出现在答案中
        ground_truths = test_dataset['response']
        unmatched_indices = []
        y_true,y_pred = [],[]
        match_count = 0
        total = len(prompt)

        for i,(q,r,gt) in enumerate(zip(prompt,results,ground_truths)):
            r_wo_q = r.replace(q, "")
            if any(word in gt.lower() for word in ['false','inactive','rejected','not approved']) and any(word in r_wo_q.lower() for word in ['false','inactive','rejected','not approved']):
                match_count += 1
                y_true.append(0)
                y_pred.append(0)
            elif any(word in gt.lower() for word in ["true", "active", "approved"]) and any(word in r_wo_q.lower() for word in ["true", "active", "approved"]):
                match_count += 1
                y_true.append(1)
                y_pred.append(1)
            else:
                match = False
                unmatched_indices.append(i)
                if gt.lower() == 'yes' or gt.lower() == 'true':
                    y_true.append(1)
                    y_pred.append(0)
                elif gt.lower() == 'no' or gt.lower() == 'false':
                    y_true.append(0)
                    y_pred.append(1)
                else:
                    raise ValueError("Ground truth must be either 'yes' or 'no'.")
            # y_true.append(1 if gt.lower() == "true" else 0)
            # y_pred.append(1 if "true" in r_wo_q.lower() else 0)
            # y_true.append(0 if any(word in gt.lower() for word in ['false','inactive','rejected','not approved']) else 1)
            # y_pred.append(0 if any(word in r_wo_q.lower() for word in ['false','inactive','rejected','not approved']) else 1)


        acc1 = match_count/total
        acc2 = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred)
        accuracy_list.append({'acc1':acc1,'acc2':acc2,'f1':f1,'roc':roc})
        # 删除当前pipeline，避免继续占用显存
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    all_results[k] = accuracy_list
    acc1_list = [x['acc1'] for x in accuracy_list]
    acc2_list = [x['acc2'] for x in accuracy_list]
    f1_list = [x['f1'] for x in accuracy_list]
    roc_list = [x['roc'] for x in accuracy_list]
    logger.info(f"Acc1 Result: {acc1_list}")
    logger.info(f"Acc2 Result: {acc2_list}")
    logger.info(f"F1 Result: {f1_list}")
    logger.info(f"ROC Result: {roc_list}")

    logger.info(f'For {k}')
    logger.info(f"Average Acc1: {statistics.mean(acc1_list):.2%}")
    logger.info(f"Average Acc2: {statistics.mean(acc2_list):.2%}")
    logger.info(f"Average F1:   {statistics.mean(f1_list):.2%}")
    logger.info(f"Average ROC:  {statistics.mean(roc_list):.2%}")

    if times > 1:
        logger.info(f"Std Dev Acc1: {statistics.stdev(acc1_list):.2%}")
        logger.info(f"Std Dev Acc2: {statistics.stdev(acc2_list):.2%}")
        logger.info(f"Std Dev F1:   {statistics.stdev(f1_list):.2%}")
        logger.info(f"Std Dev ROC:  {statistics.stdev(roc_list):.2%}")

        logger.info(f"Acc1 Result: {statistics.mean(acc1_list):.2%} ± {statistics.stdev(acc1_list):.2%}")
        logger.info(f"Acc2 Result: {statistics.mean(acc2_list):.2%} ± {statistics.stdev(acc2_list):.2%}")
        logger.info(f"F1   Result: {statistics.mean(f1_list):.2%} ± {statistics.stdev(f1_list):.2%}")
        logger.info(f"ROC  Result: {statistics.mean(roc_list):.2%} ± {statistics.stdev(roc_list):.2%}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()