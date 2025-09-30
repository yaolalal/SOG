from deepchem.molnet import load_bbbp, load_clintox, load_hiv, load_tox21, load_bace_classification
from collections import defaultdict
import torch
import dgl
import pickle
from tqdm import tqdm
from collections import deque, defaultdict
import os
from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer("all-MiniLM-L6-v2")

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

    global_node_id = num_nodes 
    node_identities[global_node_id] = "the global node which connects to all nodes"

    return node_identities

# 构建 DGL 图的辅助函数
def get_dgl_graph(adj_list, embeddings):
    src_nodes = []
    dst_nodes = []
    num_nodes = len(adj_list)
    global_node_id = num_nodes

    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            src_nodes.append(i)
            dst_nodes.append(j)
            # 添加反向边（构造无向图）
            src_nodes.append(j)
            dst_nodes.append(i)

    for i in range(num_nodes):
        src_nodes.append(global_node_id)
        dst_nodes.append(i)
        src_nodes.append(i)
        dst_nodes.append(global_node_id)

    # 转为张量
    src = torch.tensor(src_nodes)
    dst = torch.tensor(dst_nodes)

    # 构建 DGL 图
    g = dgl.graph((src, dst))
    g = dgl.to_simple(g)

    # 添加节点特征
    g.ndata['h'] = embeddings

    return g

# 通用数据加载和处理函数
def process_dataset(loader_func, dataset_name):
    print(f"Processing dataset: {dataset_name}")
    tasks, datasets, transformers = loader_func(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets
    # 打印数据集信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    graphs = defaultdict(list)

    # TEST: 观察数据集样本特征
    for split, dataset in zip(['train', 'valid', 'test'], [train_dataset, valid_dataset, test_dataset]):
        print(f"Processing split: {split}")
        for mol_graph, y, _, ids in tqdm(dataset.iterbatches(batch_size=1, deterministic=True)):
            mol = mol_graph[0]
            adj_list = mol.get_adjacency_list()
            smiles = ids[0]
            # print(f"Processing SMILES: {ids}")
            # print(f"Adjacency list: {adj_list}")
            # print(f"Labels: {y}")

            identities = get_node_identities(adj_list)
            node_texts = [0]*len(identities)  # 初始化分子文本身份列表
            for node,identity in identities.items():
                node_texts[node]=f"This is {identity}."
            assert len(node_texts) == len(adj_list)+1, "node_texts length must match adj_list length + global node num"
            embeddings = text_model.encode(node_texts, convert_to_tensor=True).cpu()   
            g = get_dgl_graph(adj_list, embeddings)
            y_temp = y[0]
            # 打印y_temp和tasks的长度
            labels = {}
            for l,task in zip(y_temp, tasks):
                labels[task] = l
            graphs[split].append({'graph': g, 'label': labels, 'text': smiles})
        print(f"y_temp length: {len(y_temp)}, tasks length: {len(tasks)}")
        
    return graphs

def main():
    # 'BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE'
    functions_map = {
        'BBBP': load_bbbp,
        'Tox21': load_tox21,
        'ClinTox': load_clintox,
        'HIV': load_hiv,
        'BACE': load_bace_classification
    }
    for dataset_name in ['BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE']:
        graphs = process_dataset(functions_map[dataset_name], dataset_name)   
        # 保存处理后的数据
        if graphs is not None:
            save_path = f"./datasets/{dataset_name}/graphs.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(graphs, f)
            print(f"Saved processed data for {dataset_name}.")
        else:
            print(f"Test...")

if __name__ == "__main__":
    main()