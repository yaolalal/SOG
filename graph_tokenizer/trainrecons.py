import os
import argparse
from utils import load_data
from codes.utils import set_seed
from codes.vq import VectorQuantize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import dgl
from dgl import batch as dgl_batch
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv
import argparse
from codes.utils import get_training_config
from tqdm import tqdm
import copy
import pandas as pd
import logging
import pickle


def setup_logger(log_file='train.log'):
    # 创建 logger 对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        # 创建 文件 handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # 创建 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 设置日志输出格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 添加 handler 到 logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

def save_metrics_to_csv(out, filename="metric_results.csv"):
    rows = []
    for split, res_list in out.items():
        for entry in res_list:
            row = entry.copy()
            row['split'] = split
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


class GraphDataset(Dataset):
    def __init__(self, graphs, device):
        self.graphs = graphs
        self.device = device

    def __len__(self):
        return len(self.graphs) 

    def __getitem__(self, idx):
        sg = self.graphs[idx]['graph']
        sg = dgl.to_simple(sg)  # 可选：移除多重边
        sg = dgl.add_self_loop(sg)
        batch_feats = sg.ndata['h']
        sg.ndata['feat'] = batch_feats

        return idx,sg.to(self.device), batch_feats.to(self.device)


def evaluate_adj_reconstruction(true_edges, pred_edges):
    
    # 输入:
    #     true_edges: torch.Tensor or np.ndarray (0/1)，shape: (N,)
    #     pred_edges: torch.Tensor or np.ndarray (0/1)，shape: (N,)
    
    # 输出:
    #     dict: 包含 accuracy, precision, recall, f1
    
    if isinstance(true_edges, torch.Tensor):
        true_edges = true_edges.cpu().numpy()
    if isinstance(pred_edges, torch.Tensor):
        pred_edges = pred_edges.cpu().numpy()

    return {
        'accuracy': accuracy_score(true_edges, pred_edges),
        'precision': precision_score(true_edges, pred_edges, zero_division=0),
        'recall': recall_score(true_edges, pred_edges, zero_division=0),
        'f1': f1_score(true_edges, pred_edges, zero_division=0)
    }


class GCN(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.num_layers = conf["num_layers"]
        self.norm_type = conf["norm_type"]
        self.dropout = nn.Dropout(conf["dropout_ratio"])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.num_layers):
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(conf["feat_dim"]))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(conf["feat_dim"]))
        input_dim = conf["feat_dim"]
        hidden_dim= input_dim
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=F.relu)
        self.graph_layer_2 = GraphConv(input_dim, input_dim, activation=F.relu)
        self.decoder_1 = nn.Linear(input_dim, hidden_dim)
        self.decoder_2 = nn.Linear(input_dim, hidden_dim)
        self.lamb_edge = conf["lamb_edge"]
        self.lamb_node = conf["lamb_node"]
        # self.lamb_node = 0

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, batched_graph, feats):
        h = feats
        h_list = []

        # 1st GCN Layer
        h = self.graph_layer_1(batched_graph, h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)
        h_list.append(h)

        # 2nd GCN Layer
        h = self.graph_layer_2(batched_graph, h)
        h_list.append(h)

        # Decode to edge & node latent space
        quantized_edge = self.decoder_1(h)
        quantized_node = self.decoder_2(h)

        # Initialize edge loss accumulator
        edge_losses = []

        # Split batched graph into list of individual graphs
        graphs = dgl.unbatch(batched_graph)
        node_offset = 0

        for g in graphs:
            num_nodes = g.num_nodes()

            # Get quantized embeddings for current graph
            qe = quantized_edge[node_offset:node_offset + num_nodes]
            node_offset += num_nodes

            # Reconstruct full predicted adjacency (logits)
            adj_pred = torch.matmul(qe, qe.T)

            # Use true adjacency from graph (dense)
            adj_true = g.adjacency_matrix().to_dense().to(feats.device)

            # Upper triangle mask for undirected graph (excluding diagonal)
            triu_mask = torch.triu(torch.ones_like(adj_true), diagonal=1).bool()
            # 只保留 i < j 的上三角区域

            adj_pred_logits = adj_pred[triu_mask]
            adj_true_triu = adj_true[triu_mask]

            # Calculate pos_weight = (# negative edges) / (# positive edges)
            # num_possible = num_nodes * (num_nodes - 1) / 2
            num_possible = num_nodes * num_nodes / 2
            num_edges = adj_true_triu.sum()
            pos_weight = (num_possible - num_edges) / (num_edges + 1e-6)

            # BCEWithLogitsLoss with pos_weight
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            edge_loss = loss_fn(adj_pred_logits, adj_true_triu)

            edge_losses.append(edge_loss)

        edge_rec_loss = self.lamb_edge * torch.stack(edge_losses).mean()

        # Optional node feature reconstruction loss
        feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)

        h_list.append(quantized_edge)

        # Final loss (alpha scaled to match previous version)
        alpha = 100
        loss = feature_rec_loss + edge_rec_loss * alpha

        return h_list, h, loss
    
@torch.no_grad()
def evaluate(model, dataloader, verbose=False):
    model.eval()
    device = next(model.parameters()).device
    total = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_loss = 0
    cnt = 0
    results = []

    for _, batched_graph, feats in dataloader:
        batched_graph = batched_graph.to(device)
        feats = feats.to(device)

        h_list, h, loss = model(batched_graph, feats)
        total_loss += loss.item()

        # 获取量化后的 edge 表征
        quantized_edge = h_list[-1]

        graphs = dgl.unbatch(batched_graph)
        node_offset = 0

        for g in graphs:
            num_nodes = g.num_nodes()

            qe = quantized_edge[node_offset:node_offset + num_nodes]
            node_offset += num_nodes

            # 预测的邻接矩阵
            adj_pred = torch.matmul(qe, qe.T)

            # 原始邻接矩阵
            adj_true = g.adjacency_matrix().to_dense().to(device)

            # 上三角区域（去掉对角线），只计算一次
            triu_mask = torch.triu(torch.ones_like(adj_true), diagonal=1).bool()

            # 提取真实和预测的边
            adj_pred_logits = adj_pred[triu_mask]
            adj_pred_labels = (torch.sigmoid(adj_pred_logits) > 0.5).int()
            adj_true_labels = adj_true[triu_mask].int()

            # 调用评估函数
            ares = evaluate_adj_reconstruction(adj_true_labels.cpu(), adj_pred_labels.cpu())
            results.append(ares)

            total += ares['accuracy']
            total_precision += ares['precision']
            total_recall += ares['recall']
            total_f1 += ares['f1']
            cnt += 1

    if verbose:
        return total / cnt, total_f1 / cnt, total_precision / cnt, total_recall / cnt

    return total_f1 / cnt, total / cnt, total_loss/len(dataloader), results

def custom_collate(batch):
    """
    batch: list of tuples: (center_node, subgraph, node_feats)
    """
    graph_idxs, subgraphs, node_feats = zip(*batch)
    batched_graph = dgl_batch(subgraphs)
    batched_feats = torch.cat(node_feats, dim=0)  # 按节点维拼接
    graph_idxs = torch.stack(graph_idxs) if isinstance(graph_idxs[0], torch.Tensor) else torch.tensor(graph_idxs)
    return graph_idxs, batched_graph, batched_feats

def train_sage(model, dataloader, optimizer, data_val, logger, lamb=10):
    model.train()
    total_loss = 0
    for step, (graph_idxs, blocks, batch_feats) in enumerate(dataloader):
        # Compute loss and prediction
        h_list, h, loss = model(blocks, batch_feats)
        # loss *= lamb
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if step % 10 == 0:
            '''print("In train_sage")
            print(model)
            print(data_val)'''
            acc,f1,precision,recall = evaluate(
                    model, data_val, verbose=True
                )
            logger.info(
                f"Step {step:3d} | loss: {loss.item():.4f} | Acc {acc:.4f} | Recall {recall:.4f} | Precision {precision:.4f} | F1 {f1:.4f}"
            ) 

        total_loss += loss.item()           

    return total_loss / len(dataloader)

def run_transductive(
        conf,
        model,
        graphs,
        optimizer,
        loss_and_score,
        logger
    ):
    set_seed(conf["seed"])
    save_dir = conf["save_dir"]
    device = conf["device"]
    batch_size = conf["batch_size"]

    dataset_train = GraphDataset(graphs['train'], device)
    data = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    logger.info(f"Train set size: {len(dataset_train)}")
    logger.info(f"Train loader size: {len(data)}")

    dataset_val = GraphDataset(graphs['valid'], device)
    data_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)  # batch_size=1 是因为每个sg结构不同
    logger.info(f"Val set size: {len(dataset_val)}")
    logger.info(f"Val loader size: {len(data_val)}")
    
    dataset_test = GraphDataset(graphs['test'], device)
    data_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)  # batch_size=1 是因为每个sg结构不同
    logger.info(f"Test set size: {len(dataset_test)}")
    logger.info(f"Test loader size: {len(data_test)}")
    
    # 这里的dataset_eval是为了评估全图采样构成所有ego-graphs的重建邻接矩阵
    dataset_eval = GraphDataset(graphs['valid']+graphs['test'], device)
    data_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)  # batch_size=1 是因为每个sg结构不同
    logger.info(f"Eval set size: {len(dataset_eval)}")
    logger.info(f"Eval loader size: {len(data_eval)}")
    
    conf['patience'] = 10000 # 刚开始关闭早停机制
    print(f'There are {conf["max_epoch"]} epochs.')

    best_acc, best_f1, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_sage(model, data, optimizer, data_val=data_val, logger=logger)

        # print(f"Epoch {epoch} | loss: {loss:.4f}")

        if epoch % conf["eval_interval"] == 0:
            f1,acc,val_loss,res_v = evaluate(
                    model, data_val,
                )
            _,_,_,res_t = evaluate(
                    model, data_test,
                )
            '''_,_,_,res = evaluate(
                    model, data,
                )'''

            logger.info(
                f"Ep {epoch:3d} | loss: {loss:.4f} | val loss: {val_loss:.4f} | acc: {acc:.4f} | f1: {f1:.4f}"
            )
            loss_and_score += [
                [
                    res_v,#仅占位
                    res_v,
                    res_t,
                ]
            ]

            if acc >= best_acc:
                best_acc = acc
                state = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir, f"model_at_epoch_{epoch}.pt")
                torch.save(state, save_path)
                logger.info(f"Model saved since acc improved at epoch {epoch}")
                count = 0
            elif f1 >= best_f1:
                best_f1 = f1
                state = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir, f"model_at_epoch_{epoch}.pt")
                torch.save(state, save_path)
                logger.info(f"Model saved since f1 improved at epoch {epoch}")
                count = 0
            elif epoch == conf["max_epoch"]:
                state = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir, f"model_at_epoch_{epoch}.pt")
                torch.save(state, save_path)
            else:
                count += 1

        # if count == conf["patience"] or epoch == conf["max_epoch"]:
        if epoch == conf["max_epoch"]:
            break

    # model.load_state_dict(state)
    # f11,acc1,_,res1 = evaluate(
    #     model, data, 
    # )
    # f12,acc2,_,res2 = evaluate(
    #     model, data_val,
    # )
    # f13,acc3,_,res3 = evaluate(
    #     model, data_test,
    # )
    # '''f14,acc4,_,res4 = evaluate(
    #     model, data_eval
    # )'''
    # logger.info(
    #     f"Train acc: {acc1:.4f} | Val acc: {acc2:.4f} | Test acc: {acc3:.4f}"
    # )
    # logger.info(
    #     f"Train f1: {f11:.4f} | Val f1: {f12:.4f} | Test f1: {f13:.4f}"
    # )
    # print(
    #     f"Train acc: {acc1:.4f} | Val acc: {acc2:.4f} | Test acc: {acc3:.4f}"
    # )
    # print(
    #     f"Train f1: {f11:.4f} | Val f1: {f12:.4f} | Test f1: {f13:.4f}"
    # )
    return 



def main():
    dataset = "all"
    log_file = f'./checkpoints/{dataset}/recons/reconstruction.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True) 
    logger = setup_logger(log_file)  # 日志文件路径
    graphs = load_data(dataset)
    args = argparse.Namespace(
        seed=0,
        dataset=dataset,
        labelrate_train=40, # 随便填的
        labelrate_val=20, # 随便填的
        split_idx=0,
    )
    args.model_config_path = "./codes/train.conf.yaml"
    args.teacher = 'GCN'
    args.dataset = 'cora'  # 这里参照先前cora数据集的设置
    args.device = 0
    conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"
    conf["device"] = device
    conf['feat_dim'] = 384
    conf['norm_type'] = "none"
    conf['codebook_size'] = 256
    conf['lamb_edge'] = 0.03
    conf['lamb_node'] = 0
    conf["learning_rate"] = 0.0001
    conf["weight_decay"] = 0.0005
    conf['dropout_ratio'] = 0
    conf["max_epoch"] = 200
    conf["batch_size"] = 512
    conf["eval_interval"] = 1
    conf["save_dir"] = f"./checkpoints/{dataset}/recons/checkpoints"
    conf['output_dir']  = f"./checkpoints/{dataset}/recons/output"
    os.makedirs(conf["save_dir"], exist_ok=True)
    os.makedirs(conf["output_dir"], exist_ok=True)

    model = GCN(conf).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=conf["weight_decay"]
    )
    loss_and_score=[]

    # feats = torch.load(f'./new_node_feats/{args.dataset}/node_to_embedding.pt')
    run_transductive(
            conf,
            model,
            graphs,
            optimizer,
            loss_and_score,
            logger
        )
    
    # 保存 loss_and_score
    score_path = os.path.join(conf['output_dir'], 'loss_and_score.pkl')
    with open(score_path, 'wb') as f:
        pickle.dump(loss_and_score, f)
    logger.info(f"Saved loss_and_score to {score_path}")


if __name__ == "__main__":
    main()