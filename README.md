# TaxoEnrich

The source code used for TaxoEnrich: Self-Supervised Taxonomy Completion via Structure-Semantic Representations [[paper]](https://arxiv.org/abs/2202.04887), published in WWW 2022.

Please cite the following work if you find the paper useful.

```
@inproceedings{jiang2022taxoenrich,
author = {Jiang, Minhao and Song, Xiangchen and Zhang, Jieyu and Han, Jiawei},
title = {TaxoEnrich: Self-Supervised Taxonomy Completion via Structure-Semantic Representations},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485447.3511935},
doi = {10.1145/3485447.3511935},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {925–934},
numpages = {10},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}	
```

Contact: Minhao Jiang (minhaoj2@illinois.edu)

## Install Guide

### Install DGL 0.4.0 version with GPU suppert using Conda

From following page: [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)

```
conda install -c dglteam dgl-cuda10.0
```

### Other packages

```
ipdb tensorboard gensim networkx tqdm more_itertools
```

## Data Preparation

For dataset used in our paper, you can directly download all input files below and skip this section.

For expanding new input taxonomies, you need to read this section and format your datasets accordingly.

[MAG_CS](https://drive.google.com/file/d/1SkH74aTW6lfmuZll06wBFwv_R1Pt6kLC/view?usp=sharing) 

[MAG_PSY](https://drive.google.com/file/d/1cTr4n65ymLneUP_9wUitNgZ1Nbl9jXZ0/view?usp=sharing) 

[SemEval_Noun](https://drive.google.com/file/d/1YdHTQp5YxLFHroF9_qvK9dGZWh8Ivi7H/view?usp=sharing) 

[SemEval_Verb](https://drive.google.com/file/d/1mdOYKHMSVT6lvYBRWROC1SJSpMxA-hK3/view?usp=sharing) 

### Step 0.a (Required): Organize your input taxonomy along with node features into the following 3 files

**1. <TAXONOMY_NAME>.terms**, each line represents one concept in the taxonomy, including its ID and surface name

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```
Since the embedding generation for training taxonomy should be separate, we require the data to be split into <TAXONOMY_NAME>.terms.train/validation/partition files beforehand.

**2. <TAXONOMY_NAME>.taxo**, each line represents one relation in the taxonomy, including the parent taxon ID and child taxon ID

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```

**3. <TAXONOMY_NAME>.terms.<EMBED_SUFFIX>.embed**, the first line indicates the vocabulary size and embedding dimension, each of the following line represents one taxon with its pretrained embeddings based on training taxonomic structures.

```
<VOCAB_SIZE> <EMBED_DIM>
taxon1_id taxon1_embedding
taxon2_id taxon2_embedding
taxon3_id taxon3_embedding
...
```

**4. <TAXONOMY_NAME>.terms.<EMBED_SUFFIX>.bertembed**, it follows the same format the embedding file above, with the pretrained embeddings based on surface names.
The embedding file follows the gensim word2vec format.

Notes:

1. Make sure the <TAXONOMY_NAME> is the same across all the 3 files.
2. The <EMBED_SUFFIX> is used to chooose what initial embedding you will use. You can leave it empty to load the file "<TAXONOMY_NAME>.terms.embed". **Make sure you can generate the embedding for a new given term.**
### Step 1: Generate the embeddings for the existing taxonomy

1. After formatting the data file as above, run

```python embedding_generation.py```

2. In the complete data, you will get 7 files representing the whole taxonomic structure and concept names as described above. And now you can generate the binary dataset file with the following step.

### Step 2: Generate the binary dataset file

1. create a folder "./data/{DATASET_NAME}"
2. put the above three required files (as well as three optional partition files) in "./data/{DATASET_NAME}"
3. under this root directory, run

```
python generate_dataset_binary.py \
    --taxon_name <TAXONOMY_NAME> \
    --data_dir <DATASET_NAME> \
    --embed_suffix <EMBED_SUFFIX> \
    --existing_partition 1 \
    --partition_pattern internal \
```

This script will first load the existing taxonomy (along with initial node features indicated by `embed_suffix`) from the previous three required files.
Then, if `existing_partition` is 0, it will generate a random train/validation/test partitions, otherwise, it will load the existing train/validation/test partition files.
Notice that if `partition_pattern` is `internal`, it will randomly sample both internal and leaf nodes for validation/test, which makes it a taxonomy completion task; if it is set `leaf`, it will become a taxonomy expansion task.
Finally, it saves the generated dataset (along with all initial node features) in one pickle file for fast loading next time.


## Model Training

### Simplest training

Write all the parameters in an config file, let's say **./config_files/config.universal.json**, and then start training.

Please check **./config_files/config.explain.json** for explanation of all parameters in config file

There are five config files under each sub dirs of **./config_files**:

1. **baselineex**: baselines for taxonomy expansion (bilinear model);
2. **tmnex**: TMN for taxonomy expansion;
3. **baseline**: baselines for taxonomy completion;
4. **tmn**: TMN for taxonomy completion;
5. **enrich**: TaxoEnrich for taxonomy completion;

```
python train.py --config config_files/config.universal.json
```

### Specifying parameters in training command

For example, you can indicate the matching method as follow:

```
python train.py --config config_files/config.universal.json --mm BIM --device 0
```

Please check **./train.py** for all configurable parameters.


### Running one-to-one matching baselines

For example, BIM method on MAG-PSY:

```
python train.py --config config_files/MAG-PSY/config.test.baseline.json --mm BIM
```

### Running Triplet Matching Network

For example, on MAG-PSY:

```
python train.py --config config_files/MAG-PSY/config.test.tmn.json --mm TMN
```

### Supporting multiple feature encoders

Although we only use initial embedding as input in our paper, our code supports combinations of complicated encoders such as both GNN and LSTM.

Check out the `mode` parameter, there are three symbols for `mode`: `r`, `p`, `g` and `s`, representing initial embedding, LSTM, GNN, and sibling encoder respectively. 

If you want to replace initial embedding with a GNN encoder, plz set `mode` to `g`; 

If you want to use a combination of initial embedding, LSTM sequential encoder and the sibling encoder, plz set `mode` to `rgs`, and then the initial embedding and embedding output by GNN and sibling encoder will be concatenated for calculating matching score; 

For GNN encoder, we defer user to Jiaming's WWW'20 paper [TaxoExpan](https://arxiv.org/abs/2001.09522);

For LSTM encoder, we collect a path from root to the anchor node, and them use LSTM to encoder it to generate representation of anchor node;

For sibling encoder, we introduce the details in the Section 3.3 in our paper.


## Model Inference

Predict on completely new taxons.

### Data Format

We assume the input taxon list is of the following format:

```
term1 \t embeddings of term1
term2 \t embeddings of term2
term3 \t embeddings of term3
...
```

The term can be either a unigram or a phrase, as long as it doesn't contain "\t" character.
The embedding is space-sperated and of the same dimension of trained model.

### Infer

```
python infer.py --resume <MODEL_CHECKPOINT.pth> --taxon <INPUT_TAXON_LIST.txt> --save <OUTPUT_RESULT.tsv> --device 0
```


### Model Organization

For all implementations, we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).
