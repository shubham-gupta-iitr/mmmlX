import json, random, time, os, base64
import numpy as np
from pprint import pprint
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

from sentence_transformers import SentenceTransformer
import torch

dataset = json.load(open("WebQA_train_val.json", "r"))

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
# make a dicts of all text entities (Q,A,Snippets) -> embeddings
import pickle
QA = []
snippets = []
QA_dict = {}
snippets_dict = {}
batch_size = 256
i=0
for k in list(dataset.keys()):
  if dataset[k]['Qcate'] == 'text':
    i += 1
    QA.append(dataset[k]['Q'])
    QA.append(dataset[k]['A'][0])
    for f in dataset[k]['txt_posFacts']:
      snippets.append(f['fact'])
    if i%batch_size==0:
      embeddings_qa = model.encode(QA)
      embeddings_snippets = model.encode(snippets)
      QA_dict.update(dict(zip(QA,embeddings_qa)))
      snippets_dict.update(dict(zip(snippets,embeddings_snippets)))
      QA = []
      snippets = []
      
    if i% 10000 == 0:
      with open("qaEmbeddings.pkl","wb") as fp:
        pickle.dump(QA_dict,fp)
      with open("snipEmbeddings.pkl","wb") as fp:
        pickle.dump(snippets_dict,fp)
    print("done writing 10000 samples")

with open("qaEmbeddings.pkl","wb") as fp:
  pickle.dump(QA_dict,fp)
with open("snipEmbeddings.pkl","wb") as fp:
  pickle.dump(snippets_dict,fp)