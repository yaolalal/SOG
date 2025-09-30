from deepchem.molnet import load_bbbp, load_clintox, load_hiv, load_tox21, load_bace_classification
from collections import defaultdict
import pickle
from tqdm import tqdm
import os


# 通用数据加载和处理函数
def process_dataset(loader_func, dataset_name):
    print(f"Processing dataset: {dataset_name}")
    tasks, datasets, transformers = loader_func(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets
    graphs = defaultdict(list)
    # 打印数据集信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # TEST: 观察数据集样本特征
    for split, dataset in zip(['train', 'valid', 'test'], [train_dataset, valid_dataset, test_dataset]):
        print(f"Processing split: {split}")
        for _, y, _, ids in tqdm(dataset.iterbatches(batch_size=1, deterministic=True)):
            smiles = ids[0]
            y_temp = y[0]
            # 打印y_temp和tasks的长度
            labels = {}
            for l,task in zip(y_temp, tasks):
                labels[task] = l
            graphs[split].append({'label': labels, 'text': smiles})
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
        attrgraphs = process_dataset(functions_map[dataset_name], dataset_name)   
        # 保存处理后的数据
        if attrgraphs is not None:
            save_path = f"./datasets/{dataset_name}/graph_attributes.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(attrgraphs, f)
            print(f"Saved processed data for {dataset_name}: {save_path}")
        else:
            print(f"Test...")

if __name__ == "__main__":
    main()