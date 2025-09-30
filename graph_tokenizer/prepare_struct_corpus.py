from scipy.spatial.distance import cdist
import os
import json
import argparse
from codes.utils import get_training_config
from trainvq import GCN
import torch
import os
import json
import random


# 加载模型
model_path = "checkpoints/all/vq/checkpoints/model_at_epoch_30.pt"
args = argparse.Namespace(
    seed=0,
    labelrate_train=40, # 随便填的
    labelrate_val=20, # 随便填的
    split_idx=0, # 随便填的
)
args.model_config_path = "./codes/train.conf.yaml"
args.teacher = 'GCN'
args.dataset = 'cora'
args.device = 0
conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
conf = dict(args.__dict__, **conf)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"
conf["device"] = device
conf['feat_dim'] = 384
conf['codebook_size'] = 256
conf['lamb_edge'] = 0.03
conf['lamb_node'] = 0
conf["weight_decay"] = 0.0005
conf['dropout_ratio'] = 0
conf["norm_type"] = "none"
model = GCN(conf).to(device)

load_state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = model.state_dict()
model_state_dict.update(load_state_dict)
model.load_state_dict(model_state_dict)
model.eval()
print(model)


# 得到codebook
codebook = model.vq._codebook.embed.squeeze()  # shape: [N, D]
# 如果codebook.shape不是N * D 报错
assert codebook.shape[0] == conf['codebook_size'] and codebook.shape[1] == conf['feat_dim'], "Codebook shape is not correct"

# 得到相似度排序
codes = codebook.cpu().detach().numpy()  # shape: [N, D]
D = cdist(codes, codes, metric="cosine")  # 或 metric='euclidean'


# 🌟 构造结构对比学习语料
# 构造正负样本对：遍历所有相似度对，将小于0.2的相似度对添加到相似对列表中，大于0.8的相似度对添加到不相似对列表中
similar_pairs = []
dissimilar_pairs = []
for i in range(len(codes)):
    for j in range(i+1, len(codes)):
        d = D[i][j]
        if d < 0.2:
            similar_pairs.append((i, j))
        elif d > 0.8:
            dissimilar_pairs.append((i, j))
# 添加进语料模版
examples = []
# 添加正样本对
for a, b in similar_pairs:
    examples.append({
        "instruction": (
            f"You are given two structure tokens: <gstruct_{a}> and <gstruct_{b}>.\n"
            f"Each structure token represents a unique graph pattern (e.g., a molecular graph).\n"
            f"Based on their structure semantics, determine whether these two tokens likely represent similar graph topologies or not.\n"
            f"Explain your reasoning briefly."
        ),
        "output": (
            f"Yes, the structure tokens <gstruct_{a}> and <gstruct_{b}> represent highly similar graph patterns. "
            f"Their embeddings lie close to each other in the structure representation space, "
            f"indicating similar graph topology, such as node degrees, neighborhood distribution, and structural roles."
        )
    })
# 添加负样本对
for a, b in dissimilar_pairs:
    examples.append({
        "instruction": (
            f"You are given two structure tokens: <gstruct_{a}> and <gstruct_{b}>.\n"
            f"Each structure token represents a distinct graph from a larger graph distribution. "
            f"Evaluate their structural similarity and explain your conclusion."
        ),
        "output": (
            f"No, the structure tokens <gstruct_{a}> and <gstruct_{b}> represent different types of graph patterns. "
            f"Their embeddings are distant in the structure representation space, reflecting distinct topological features or motif patterns."
        )
    })
# 🔀 打乱所有样本
random.shuffle(examples)

# ✍️写入文件
file_name = "./corpus/struct_token_similarity.jsonl"
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, "w") as f:
    for example in examples:
        json.dump(example, f)
        f.write("\n")
print(f"Distance corpus saved to {file_name}")



# 🌟 构造KNN相似结构语料
TEMPLATES = [
    "Starting from <gstruct_{}>, the 5 most similar structure tokens in the graph embedding space are: {}.",
    "Structure token <gstruct_{}> is closest to the following tokens based on structural similarity: {}.",
    "Starting from <gstruct_{}>, the 5 most dissimilar structure tokens in the graph embedding space are: {}.",
    "Structure token <gstruct_{}> is farthest from the following tokens in structure space: {}.",
]

def generate_struct_corpus_topk(D, code_prefix="gstruct_", output_file="struct_code_train_corpus.txt", k=5):
    N = D.shape[0]
    corpus = []

    center_ids = list(range(N))
    random.shuffle(center_ids)  # 打乱 center_id 的顺序

    for center_id in center_ids:
        distances = [(j, D[center_id][j]) for j in range(N) if j != center_id]
        sorted_by_distance = sorted(distances, key=lambda x: x[1])
        nearest = [f"<{code_prefix}{j}>" for j, _ in sorted_by_distance[:k]]
        farthest = [f"<{code_prefix}{j}>" for j, _ in sorted_by_distance[-k:]]

        # 加入相近模板
        corpus.append(TEMPLATES[0].format(center_id, " ".join(nearest)))
        # corpus.append(TEMPLATES[1].format(center_id, " ".join(nearest)))

        # 加入不相近模板
        corpus.append(TEMPLATES[2].format(center_id, " ".join(farthest)))
        # corpus.append(TEMPLATES[3].format(center_id, " ".join(farthest)))

    for center_id in center_ids:
        distances = [(j, D[center_id][j]) for j in range(N) if j != center_id]
        sorted_by_distance = sorted(distances, key=lambda x: x[1])
        nearest = [f"<{code_prefix}{j}>" for j, _ in sorted_by_distance[:k]]
        farthest = [f"<{code_prefix}{j}>" for j, _ in sorted_by_distance[-k:]]

        # 加入相近模板
        # corpus.append(TEMPLATES[0].format(center_id, " ".join(nearest)))
        corpus.append(TEMPLATES[1].format(center_id, " ".join(nearest)))

        # 加入不相近模板
        # corpus.append(TEMPLATES[2].format(center_id, " ".join(farthest)))
        corpus.append(TEMPLATES[3].format(center_id, " ".join(farthest)))

    with open(output_file, "w") as f:
        for line in corpus:
            f.write(line + "\n")
    
    print(f"✅ Saved {len(corpus)} training samples to {output_file}")
# ✍️写入文件
output_file = "./corpus/struct_code_train_corpus.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
generate_struct_corpus_topk(D, code_prefix="gstruct_", output_file=output_file, k=5)


# 🌟 构造结构描述关联语料
