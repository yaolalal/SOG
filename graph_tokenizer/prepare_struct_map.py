from trainvq import evaluate,GCN,GraphDataset,custom_collate
from torch.utils.data import DataLoader
from utils import load_data
import torch
import argparse
from codes.utils import get_training_config
import os
from collections import deque, defaultdict
from deepchem.molnet import load_bbbp,load_tox21, load_clintox, load_hiv, load_bace_classification
from tqdm import tqdm
import json

batch_size = 128
dataset = "all"  # or "Tox21", "ClinTox", "HIV", "BACE",
model_path = "checkpoints/all/vq/checkpoints/model_at_epoch_30.pt"

# 加载数据集
graphs = load_data(dataset)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"
dataset_train = GraphDataset(graphs['train'], device)
data = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
dataset_val = GraphDataset(graphs['valid'], device)
data_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)  
dataset_test = GraphDataset(graphs['test'], device)
data_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate) 
# 打印数据集和数据加载器的长度
print(f"Length of training dataset: {len(dataset_train)}")
print(f"Length of validation dataset: {len(dataset_val)}")
print(f"Length of test dataset: {len(dataset_test)}")   
print(f"Length of training dataloader: {len(data)}")
print(f"Length of validation dataloader: {len(data_val)}")
print(f"Length of test dataloader: {len(data_test)}")


# 加载模型
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

# 评估模型：找到和每个graph对应的structural code
@torch.no_grad()
def evaluate_codes(model, dataloader, verbose=True):
    model.eval() 
    codes_map = {}
    print("Evaluating...")
    for ind,(graph_inds, batched_graph, feats) in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        feats = feats.to(device)
        h_list, quantized, loss, detailed_loss, codebook, dist_metric_list = model(batched_graph, feats)
        num_codes = detailed_loss['num_codes']
        graph_ind = [graph_inds[0].item() if isinstance(graph_inds[0], torch.Tensor) else graph_inds[0]][0]
        codes_map[graph_ind] = torch.argmax(dist_metric_list[0], dim=1).tolist()
        if verbose:
            print(f'num of codes used in {ind} batch: {num_codes:.1f}')
    return codes_map
data = DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=custom_collate)
data_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=custom_collate)
data_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=custom_collate)
codes_map_train = evaluate_codes(model, data)
codes_map_val = evaluate_codes(model, data_val)
codes_map_test = evaluate_codes(model, data_test)
# 打印结果:训练/验证/测试集长度
print(len(codes_map_train),len(codes_map_val),len(codes_map_test))
# 打印结果：训练集第一个graph的长度和最后一个节点（global node）对应的code
print(len(codes_map_train[0]),codes_map_train[0][-1])

training_graph_codes_map = {}
validation_graph_codes_map = {}
test_graph_codes_map = {}
for i in range(len(codes_map_train)):
    training_graph_codes_map[i] = codes_map_train[i][-1]  # 只保留每个图的最后一个节点(全局节点)的代码
for i in range(len(codes_map_val)):
    validation_graph_codes_map[i] = codes_map_val[i][-1]  # 只保留每个图的最后一个节点(全局节点)的代码
for i in range(len(codes_map_test)):
    test_graph_codes_map[i] = codes_map_test[i][-1]  # 只保留每个图的最后一个节点(全局节点)的代码
codes_map = {
    "train": training_graph_codes_map,
    "valid": validation_graph_codes_map,
    "test": test_graph_codes_map
}

save_path = f"datasets/{dataset}/codes_map.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
torch.save(codes_map, save_path)



# 🌟🌟🌟 构造结构描述关联语料 🌟🌟🌟
# codes_map = torch.load("datasets/all/codes_map.pt")
def get_node_identities(adj_list, max_hop=None):
    num_nodes = len(adj_list)
    
    # Step 1: 计算每个节点的度
    degrees = [len(neigh) for neigh in adj_list]
    
    # Step 2: 找到度最大的中心节点
    center_node = max(range(num_nodes), key=lambda i: degrees[i])
    
    # Step 3: BFS 获取每个节点的 hop 数
    node_to_hop = {center_node: 0}
    queue = deque([(center_node, 0)])
    visited = set([center_node])
    
    while queue:
        node, hop = queue.popleft()
        if max_hop is not None and hop >= max_hop:
            continue
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                node_to_hop[neighbor] = hop + 1
                queue.append((neighbor, hop + 1))   
    if max_hop is None:           
        max_hop = max(node_to_hop.values())

    # Step 4: 为每一跳的节点按照度排序，赋予“第 s 个邻居”位置
    hop_to_nodes = defaultdict(list)
    for node, hop in node_to_hop.items():
        if hop > 0:
            hop_to_nodes[hop].append(node)
    
    # 生成最终身份映射
    node_identities = {}
    node_identities[center_node] = "the center node"

    for h in range(1, max_hop + 1):
        sorted_nodes = sorted(hop_to_nodes[h], key=lambda n: -degrees[n])
        for s, node in enumerate(sorted_nodes):
            node_identities[node] = f"{h}-hop neighbor {s+1}"

    # 剩下的节点（不在 hop 范围内）
    for node in range(num_nodes):
        if node not in node_identities:
            node_identities[node] = "unreachable"

    return node_identities
def get_identity_to_node_mapping(adj_list, max_hop=None):
    node_to_identity = get_node_identities(adj_list, max_hop)
    
    # identity → node_id 的反向映射（非唯一，故使用列表）
    identity_to_node = {}

    for node_id, identity in node_to_identity.items():
        identity_to_node[identity] = node_id  # identity是唯一的，直接反转

    # 自定义排序函数
    def identity_sort_key(identity):
        if identity == "the center node":
            return (0, 0)
        elif identity == "unreachable":
            return (float('inf'), float('inf'))
        else:
            # 格式: "h-hop neighbor s"
            parts = identity.split()
            hop = int(parts[0].split('-')[0])
            count = int(parts[-1])
            return (hop, count)
    
    # 排序后的 (identity, node_id) 列表
    sorted_items = sorted(identity_to_node.items(), key=lambda x: identity_sort_key(x[0]))
    return sorted_items  # 返回 (identity, node_id) 的有序列表

splits = ['train', 'valid', 'test']
graph_descs = {split: [] for split in splits}
sentence_template = '{src_node_identity} and {dst_node_identity} are connected.\n'

for dataset_name in ['BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE']:
    functions_map = {
        'BBBP': load_bbbp,
        'Tox21': load_tox21,
        'ClinTox': load_clintox,
        'HIV': load_hiv,
        'BACE': load_bace_classification
    }
    load_function = functions_map.get(dataset_name)
    if not load_function:
        raise ValueError(f"Dataset {dataset_name} is not supported. Available datasets: {list(functions_map.keys())}")
    tasks, datasets, transformers = load_function(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

    for dataset,key in zip([train_dataset, valid_dataset, test_dataset],splits):
        print(f"Processing dataset {dataset_name}-{key}({len(dataset)})...")
        for X, y, _, ids in tqdm(dataset.iterbatches(batch_size=1, deterministic=True),desc=f"Processing {key} dataset"):
            # print("X =", X)   # array[分子图对象（GraphConvMol）]
            # print("y =", y)   # 标签
            # print("id =", ids) # 样本 ID：smiles字符串
            mol_graph = X[0]  # 注意：X 是一个 list
            adj_list = mol_graph.get_adjacency_list()  # List[List[int]]
            graph_desc = ''
            node2identity_map = get_node_identities(adj_list)
            identity2node_map = get_identity_to_node_mapping(adj_list)
            for identity, node_id in identity2node_map:
                src_node_identity = identity
                for neigh_id in adj_list[node_id]:
                    dst_node_identity = node2identity_map[neigh_id]
                    sentence = sentence_template.format(src_node_identity=src_node_identity, dst_node_identity=dst_node_identity)
                    graph_desc += sentence
            graph_desc += f'A global node is added to represent the entire graph.'
            graph_descs[key].append(graph_desc)

final_result = {split: [] for split in splits}
for split in splits:
    assert len(codes_map[split]) == len(graph_descs[split]), f"Length mismatch for {split} split: {len(codes_map[split])} vs {len(graph_descs[split])}"
    for code, desc in zip(list(codes_map[split].values()), graph_descs[split]):
        final_result[split].append({'code': code, 'desc': desc})
        
# Save the final result
output_path = f"./corpus/graph_desc_token_pairs.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# 存储final_result到json文件
with open(output_path, 'w') as f:
    json.dump(final_result, f)
print(f"Graph description-token pairs saved to {output_path}")


