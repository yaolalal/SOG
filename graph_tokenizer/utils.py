import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

def load_all_datasets():
    print("Loading and balancing all datasets...")
    datasets_to_load = ['BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE']
    splits = ['train', 'valid', 'test']
    graphs_by_dataset = defaultdict(dict)  # 每个数据集对应一个三分图集

    # 1. 加载所有数据集
    for dname in datasets_to_load:
        with open(f'./datasets/{dname}/graphs.pkl', 'rb') as f:
            g = pickle.load(f)
            for split in splits:
                graphs_by_dataset[dname][split] = g[split]

    # 2. 对 train 做平衡（valid 和 test 通常不需要复制）
    max_train_len = max(len(graphs_by_dataset[d]['train']) for d in datasets_to_load)
    print(f"Max train length across datasets: {max_train_len}")

    # 3. 合并所有图结构，复制 train 以平衡数据量
    graphs_all = {split: [] for split in splits}
    for dname in datasets_to_load:
        for split in splits:
            graphs = graphs_by_dataset[dname][split]
            if split == 'train':
                # repeat_factor = math.ceil(max_train_len / len(graphs)) if len(graphs) > 0 else 1
                # balanced_graphs = (graphs * repeat_factor)[:max_train_len]  # 切到最长长度
                # graphs_all[split].extend(balanced_graphs)
                graphs_all[split].extend(graphs)
            else:
                graphs_all[split].extend(graphs)  # valid/test 不重复
    return graphs_all


def load_data(dataset):
    if dataset == 'all':
        graphs_all = load_all_datasets()
        return graphs_all
    else:
        file_path = f'datasets/{dataset}/graphs.pkl'
        with open(file_path, 'rb') as f:
            graphs = pickle.load(f)
        return graphs


def summarize_and_plot_metrics(metric_list):
    """
    输入是一个由 metric dict 构成的列表：
    [{'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}, ...]

    输出每个指标的均值、方差、最小值、最大值，并进行可视化。
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    stats = {m: {} for m in metrics}
    values = {m: [] for m in metrics}

    # 收集每个指标的值
    for entry in metric_list:
        for m in metrics:
            values[m].append(entry[m])

    # 计算统计量
    for m in metrics:
        arr = np.array(values[m])
        stats[m]['mean'] = np.mean(arr)
        stats[m]['std'] = np.std(arr)
        stats[m]['min'] = np.min(arr)
        stats[m]['max'] = np.max(arr)

    # 打印
    print("\n📊 Metric Summary:")
    for m in metrics:
        print(f"{m.capitalize():<10} | Mean: {stats[m]['mean']:.4f} | Std: {stats[m]['std']:.4f} | "
              f"Min: {stats[m]['min']:.4f} | Max: {stats[m]['max']:.4f}")

    # 可视化
    plt.figure(figsize=(10, 4))
    for i, m in enumerate(metrics):
        plt.subplot(1, 4, i + 1)
        plt.hist(values[m], bins=50, color='skyblue', edgecolor='black')
        plt.title(m.capitalize())
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.grid(True)

    plt.suptitle("Metric Distribution Across Graphs")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return stats

