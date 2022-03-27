import torch
import networkx as nx
from tqdm import tqdm
from gensim.models.wrappers import FastText
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import dgl
from gensim.models import KeyedVectors
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import torch
from transformers import *
from collections import defaultdict, deque
from typing import List, Optional
import pickle
import networkx as nx
from networkx.algorithms import descendants, ancestors
import dgl
from gensim.models import KeyedVectors
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import pickle
import time
from tqdm import tqdm
import random
import copy
from itertools import chain, product, combinations
import os
import multiprocessing as mp
from functools import partial
from collections import defaultdict, deque
import more_itertools as mit


def _get_holdout_subgraph(g_full, node_ids):
    full_graph = g_full.to_networkx()
    node_to_remove = [n for n in full_graph.nodes if n not in node_ids]
    subgraph = full_graph.subgraph(node_ids).copy()
    for node in node_to_remove:
        parents = set()
        children = set()
        ps = deque(full_graph.predecessors(node))
        cs = deque(full_graph.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(full_graph.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(full_graph.successors(c))
        for p in parents:
            for c in children:
                subgraph.add_edge(p, c)
        # remove jump edges
    node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
    for node in subgraph.nodes():
        if subgraph.out_degree(node) > 1:
            successors1 = set(subgraph.successors(node))
            successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
            checkset = successors1.intersection(successors2)
            if checkset:
                for s in checkset:
                    subgraph.remove_edge(node, s)
    return subgraph


def get_root(graph):
    return list(nx.topological_sort(graph))[0]


def get_parents(graph, node):
    return [edge[0] for edge in graph.in_edges(node)]


def get_children(graph, node):
    return [edge[1] for edge in graph.out_edges(node)]


def get_siblings_with_most_k(graph, node, k=5):
    parents = get_parents(graph, node)
    siblings = []
    for p in parents:
        if p == node:
            continue
        siblings += get_children(graph, p)
    if k > len(siblings):
        k = len(siblings)
    return random.sample(siblings, k)


def get_leaf(graph):
    return [node for node in graph if graph.out_degree(node) == 0]


def _get_path_to_root(graph):
    node2root_path = {n: [] for n in graph.nodes}
    q = deque([get_root(graph)])
    node2root_path[get_root(graph)] = [[get_root(graph)]]
    visit = []
    while q:
        i = q.popleft()
        if i in visit:
            continue
        else:
            visit.append(i)
        children = get_children(graph, i)
        for c in children:
            if c not in q:
                q.append(c)
            for path in node2root_path[i]:
                node2root_path[c].append([c] + path)
    return node2root_path


def _get_path_to_leaf(graph):
    leafs = [n for n in graph.nodes if graph.out_degree(n) == 0]
    node2leaf_path = {n: [] for n in graph.nodes}
    q = deque(leafs)
    for n in leafs:
        node2leaf_path[n] = [[n]]
    visit = []
    while q:
        i = q.popleft()
        if i in visit:
            continue
        else:
            visit.append(i)
        parents = get_parents(graph, i)
        for p in parents:
            if p not in q:
                q.append(p)
            for path in node2leaf_path[i]:
                node2leaf_path[p].append([p] + path)
    return node2leaf_path


def generate_sentences(root_path, leaf_path, graph, node, taxoord2term):
    parents = root_path[node]
    children = leaf_path[node]
    # siblings = get_siblings_with_most_k(graph, node, k=5)
    parent_sentence = []
    children_sentence = []
    sibling_sentence = []
    for i in range(len(parents)):
        path = parents[i]
        ascendants = path[1:]
        if len(ascendants) == 0:
            continue
        sentence = ""
        for j in range(len(ascendants)):
            if j != len(ascendants) - 1:
                sentence += taxoord2term[ascendants[j]] + ", "
            else:
                sentence += taxoord2term[ascendants[j]]
                sentence += " is a super class of {}".format(taxoord2term[node])
        parent_sentence.append(sentence)
    for i in range(len(children)):
        path = children[i]
        descendants = path[1:]
        if len(descendants) == 0:
            continue
        sentence = ""
        for j in range(len(descendants)):
            if j == len(descendants) - 1:
                sentence += taxoord2term[descendants[j]]
                sentence += " is a subclass of {}".format(taxoord2term[node])
            else:
                sentence += taxoord2term[descendants[j]] + ", "
        children_sentence.append(sentence)
    # for i in range(len(siblings)):
    #     sibling_sentence.append('{} is a sibling of {}'.format(taxoid2term[siblings[i]], taxoid2term[node]))
    return parent_sentence, children_sentence


def emb_extract(sentence, key_words, device, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)], device=device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    token_list_new = list()
    idx_new2old_map = list()
    n = 0
    for i, token in enumerate(tokens):
        if '##' not in token:
            token_list_new.append(token)
            idx_new2old_map.append([i + 1])
            n += 1
        else:
            token_list_new[n - 1] += token.replace('##', '')
            idx_new2old_map[n - 1].append(i + 1)
    emb_list = list()
    for tgt in key_words:
        target = ''.join([i.replace("##", '') for i in tokenizer.tokenize(tgt)])
        idx = token_list_new.index(target)
        old_idx = idx_new2old_map[idx]
        embs = last_hidden_states[0, old_idx, :].sum(dim=0) / len(old_idx)  # average emb of all tokens
        emb_list.append(embs.to('cpu'))
    return emb_list


def phrase_emb_extract(sentence: str, phrase_list: List[str], device='cuda:0') -> List[torch.tensor]:
    emb_list = list()
    for phrase in phrase_list:
        words = phrase.split(' ')
        embs = emb_extract(sentence, words, device)
        phrase_emb = torch.stack(embs).sum(dim=0) / len(embs)
        emb_list.append(phrase_emb)
    return emb_list[0]


def generate_embeddings(taxoid2sent, node, taxoord2term):
    parent_sentence, children_sentence = taxoid2sent[node][0], taxoid2sent[node][1]
    if len(parent_sentence) == 0:
        parent_embeddings = torch.zeros(768)
    else:
        parent_embeddings = torch.mean(
            torch.stack([phrase_emb_extract(i, [taxoord2term[node]]) for i in parent_sentence]), dim=0)
    if len(children_sentence) == 0:
        children_embeddings = torch.zeros(768)
    else:
        children_embeddings = torch.mean(
            torch.stack([phrase_emb_extract(i, [taxoord2term[node]]) for i in children_sentence]), dim=0)
    # if len(sibling_sentence) == 0:
    #     sibling_embeddings = torch.zeros(768)
    # else:
    #     sibling_embeddings = torch.mean(torch.stack([phrase_emb_extract(i, [taxoid2term[node]]) for i in sibling_sentence]), dim=0)
    if len(parent_sentence) == 0 and len(children_sentence) != 0:
        res = torch.mean(torch.stack([children_embeddings]), dim=0)
    elif len(parent_sentence) != 0 and len(children_sentence) == 0:
        res = torch.mean(torch.stack([parent_embeddings]), dim=0)
    elif len(parent_sentence) != 0 and len(children_sentence) != 0:
        res = torch.mean(torch.stack([parent_embeddings, children_embeddings]), dim=0)
    else:
        res = torch.zeros(768)
    return res


def extract_bert_embedding(sentence, key_words, model, tokenizer, device='cuda:3'):
    tokens = tokenizer.tokenize(sentence)
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)], device=device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0][0]
    res = torch.mean(last_hidden_states, dim=0)
    return res


def main():
    with open('./data/MAG_CS/computer_science.pickle.bin', 'rb') as f:
        data = pickle.load(f)
        g_full = data['g_full']
        name = data['name']
        vocab = data["vocab"]
        train_node_ids = data["train_node_ids"]
        validation_node_ids = data["validation_node_ids"]
        test_node_ids = data["test_node_ids"]
    taxoid2term = {}
    taxoterm2id = {}
    taxoid2ord = {}
    taxoord2id = {}
    taxoord2term = {}
    ord = 0
    with open('./data/MAG_CS/computer_science.terms', 'r') as f:
        for lines in f:
            line = lines.strip()
            if line:
                taxo_id = line.split('\t')[0]
                taxo_term = line.split('\t')[1].split('||')[0]
                if '_' in taxo_term:
                    taxo_term = taxo_term.replace('_', ' ')
                if '-' in taxo_term:
                    taxo_term = taxo_term.replace('-', ' ')
                if "'" in taxo_term:
                    taxo_term = taxo_term.replace("'", '')
                if "/" in taxo_term:
                    taxo_term = taxo_term.replace("/", " ")
                taxoid2term[taxo_id] = taxo_term
                taxoterm2id[taxo_term] = taxo_id
                taxoid2ord[taxo_id] = ord
                taxoord2id[ord] = taxo_id
                taxoord2term[ord] = taxo_term
                ord += 1
    taxonomy = nx.DiGraph()
    with open('./data/MAG_CS/computer_science.taxo', 'r') as f:
        for lines in f:
            line = lines.strip()
            if line:
                parent = line.split('\t')[0].split('||')[0]
                child = line.split('\t')[1].split('||')[0]
                if '_' in taxo_term:
                    taxo_term = taxo_term.replace('_', ' ')
                if '-' in taxo_term:
                    taxo_term = taxo_term.replace('-', ' ')
                if "'" in taxo_term:
                    taxo_term = taxo_term.replace("'", '')
                if "/" in taxo_term:
                    taxo_term = taxo_term.replace("/", " ")
                taxonomy.add_edge(parent, child)
    for node in taxoid2term:
        if node not in taxonomy.nodes:
            taxonomy.add_node(node)
    print(taxonomy.number_of_nodes())
    print(taxonomy.number_of_edges())

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    core_subgraph = _get_holdout_subgraph(train_node_ids)
    root_path = _get_path_to_root(core_subgraph)
    leaf_path = _get_path_to_leaf(core_subgraph)

    taxoid2sent = {}
    nodes = list(core_subgraph.nodes)
    for i in tqdm(range(len(nodes)), desc="Generating sentences"):
        parent_sentence, children_sentence = generate_sentences(root_path, leaf_path, core_subgraph, nodes[i],
                                                                taxoord2term)
        taxoid2sent[nodes[i]] = [parent_sentence, children_sentence]
    count = 0
    for key in taxoid2sent:
        count += len(taxoid2sent[key][0]) + len(taxoid2sent[key][1])
    print("Total number of sentences: ", count)
    taxoid2emb = {}
    nodes = list(taxonomy.nodes)
    for i in tqdm(range(len(nodes)), desc="Loading embeddings"):
        id = taxoord2id[nodes[i]]
        taxoid2emb[id] = generate_embeddings(taxoid2sent, nodes[i], taxoord2term)

    with open('./data/MAG_CS/computer_science.terms.embed', 'w') as f:
        f.write(f"{len(taxoid2emb)} 768\n")
        for ele in sorted(taxoid2emb.items(), key=lambda x: x[0]):
            embed_string = " ".join([str(a.item()) for a in ele[1]])
            f.write(f"{ele[0]} {embed_string}\n")

    taxoid2bertemb = {}
    for node in tqdm(taxonomy.nodes):
        name = taxoid2term[node]
        emb = extract_bert_embedding(name, name, model, tokenizer)
        taxoid2bertemb[node] = emb

    with open('./data/MAG_CS/computer_science.terms.bertembed', 'w') as f:
        f.write(f"{len(taxoid2bertemb)} 768\n")
        for ele in sorted(taxoid2bertemb.items(), key=lambda x: x[0]):
            embed_string = " ".join([str(a.item()) for a in ele[1]])
            f.write(f"{ele[0]} {embed_string}\n")

    with open('./data/MAG_CS/computer_science.terms.train', 'w') as f:
        for node in train_node_ids:
            id = taxoord2id[node]
            f.write(f"{id}\t{taxoid2term[id]}\n")

    with open('./data/MAG_CS/computer_science.terms.validation', 'w') as f:
        for node in validation_node_ids:
            id = taxoord2id[node]
            f.write(f"{id}\t{taxoid2term[id]}\n")

    with open('./data/MAG_CS/computer_science.terms.test', 'w') as f:
        for node in test_node_ids:
            id = taxoord2id[node]
            f.write(f"{id}\t{taxoid2term[id]}\n")


if __name__ == "__main__":
    main()
