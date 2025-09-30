from trainvq import evaluate,GCN,GraphDataset,custom_collate
from torch.utils.data import DataLoader
from utils import load_data
import torch
import argparse
from codes.utils import get_training_config
import os

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


@torch.no_grad()
def mappings(model, dataloader, verbose=True):
    model.eval() 
    h_collections = {}
    print("Evaluating...")
    for ind,(graph_inds, batched_graph, feats) in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        feats = feats.to(device)
        h_list, quantized, loss, detailed_loss, codebook, dist_metric_list = model(batched_graph, feats)
        h_need = h_list[1]
        num_codes = detailed_loss['num_codes']
        graph_ind = [graph_inds[0].item() if isinstance(graph_inds[0], torch.Tensor) else graph_inds[0]][0]
        h_collections[graph_ind] = h_need.cpu().detach()
        if verbose:
            print(f'num of codes used in {ind} batch: {num_codes:.1f}')
    return h_collections
data = DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=custom_collate)
data_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=custom_collate)
data_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=custom_collate)
h_collections_train = mappings(model, data)
h_collections_val = mappings(model, data_val)
h_collections_test = mappings(model, data_test)

training_graph_h_collections = {}
validation_graph_h_collections = {}
test_graph_h_collections = {}
for i in range(len(h_collections_train)):
    training_graph_h_collections[i] = h_collections_train[i][-1]  # 只保留每个图的最后一个节点(全局节点)的代码
for i in range(len(h_collections_val)):
    validation_graph_h_collections[i] = h_collections_val[i][-1]  # 只保留每个图的最后一个节点(全局节点)的代码
for i in range(len(h_collections_test)):
    test_graph_h_collections[i] = h_collections_test[i][-1]  # 只保留每个图的最后一个节点(全局节点)的代码

def dict_to_matrix(embedding_dict):
    """
    将字典形式的图 embedding 转化为矩阵 tensor
    embedding_dict: {idx: tensor(hidden_dim)}
    返回: tensor(num_graphs, hidden_dim)
    """
    # 按索引顺序取值
    emb_list = [embedding_dict[i] for i in range(len(embedding_dict))]
    # 堆叠成矩阵
    matrix = torch.stack(emb_list, dim=0)
    return matrix

training_graph_h = dict_to_matrix(training_graph_h_collections)
validation_graph_h = dict_to_matrix(validation_graph_h_collections)
test_graph_h = dict_to_matrix(test_graph_h_collections)

print("training_graph_h shape:", training_graph_h.shape)
print("validation_graph_h shape:", validation_graph_h.shape)
print("test_graph_h shape:", test_graph_h.shape)


# 保存
torch.save(training_graph_h, "datasets/all/training_graph_h.pt")
torch.save(validation_graph_h, "datasets/all/valid_graph_h.pt")
torch.save(test_graph_h, "datasets/all/test_graph_h.pt")

# 加载示例
# training_graph_h_loaded = torch.load("training_graph_h.pt")
