# <SOGₖ> : One LLM Token for Explicit Graph Structural Understanding
The repository for paper "<SOGₖ> : One LLM Token for Explicit Graph Structural Understanding"

![image](https://github.com/yaolalal/SOG/blob/main/system_model.png)

## Step 1: Constructing structural tokenizer

### Data preparing

Before training, entering the ./graph_tokenizer and run attr_preprocess.py and data_preprocess.py to generate the contents of ./graph_tokenizer/datasets.

```
├── datasets
│   ├── BACE
│   │   ├── graph_attributes.pkl
│   │   ├── graphs.pkl
│   ├── BBBP
│   │   ├── graph_attributes.pkl
│   │   ├── graphs.pkl
│   ├── HIV
│   │   ├── graph_attributes.pkl
│   │   ├── graphs.pkl
│   ├── ClinTox
│   │   ├── graph_attributes.pkl
│   │   ├── graphs.pkl
│   ├── Tox21
│   │   ├── graph_attributes.pkl
│   │   ├── graphs.pkl
│   
```

### Training Structural Tokenizer

Run trainrecons.py to train the GCN encoder and MLP decoder on structural reconstruction task. Using the best checkpoint as initialization, run trainvq.py to train the graph embeddings of structural vocabulary.

```
python trainrecons.py
# run Evaluation.ipynb to find the epoch of the best checkpoint

# specify 'best_epoch' as an integer
python trainvq.py --epochs 60 --freeze true --dataset all --loadepoch best_epoch 
```

### Evaluation of Structural Tokenizer
Run Evaluation.ipynb.

### Preparing hybrid QAs corpora
```
python prepare_struct_map.py
python prepare_struct_corpus.py
```
After that, the ./graph_tokenizer/corpus will include three components: graph_desc_token_pairs.json, struct_code_train_corpus.txt, struct_token_similarity.jsonl. They are used to fill templates and form **Description-Token Pairs Matching**, **k-Nearest Token Neighbour Matching**, **True/False Structure Similarity Judgment** corpora correspondingly. Copy it to ./LLM_tuning.

## Step 2: Token alignment with hybrid QAs

### Data preparing
Entering ./LLM_tuning, run Data Preprocess.ipynb.

### Token alignment
Run pretune1stage.py and pretune2stage.py sequentially.
In the first stage, we only tune the LLM embeddings of newly added structural tokens;
In the second stage, we tune both the LLM embeddings of newly added tokens and the backbone of LLM.
```
python pretune1stage.py
python pretune2stage.py
```

## Step 3: Downstream application
Finetune LLM for specific downstream tasks.
```
# specify united_list as a list of task names to be trained jointly
# choose different data balance ratio to construct ds_dict
# customize data_prefix based on the data used and it will be used to name the trained models
python sfttune.py --data_path corpus/ds_dict.pkl --data_prefix ori --num_train_epochs 3 --united_list united_list
```

Evaluate downstream tasks performance.
```
# specify test_data path (can be provided as { 'test': dataset } or simply as the dataset.)
# set times to number of repeated evaluations
# specify checkpoint_paths as LLM checkpoints fine-tuned under each task
python eval_all.py --data_path corpus/ds_dict.pkl --times 3 --checkpoint_paths checkpoint_paths
```
