import json, random, time, os, base64
import numpy as np
from pprint import pprint
import pickle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
model = model.cuda()
dataset = json.load(open("WebQA_train_val.json", "r"))

save_path  = "Retreival/bert/ImgQueries_embeddings.pkl"
txt = []
text_keys=[]
keys = []
emb_dict={}
batch_size = 200
i=0
for k in list(dataset.keys()):
  if dataset[k]['Qcate'] != 'text':
      text_keys.append(k)
print(len(text_keys))
for q_indx, k in enumerate(text_keys):
    keys.append(k)
    txt.append(dataset[k]['Q'])
    if k not in emb_dict:
        emb_dict[k]= {}
        emb_dict[k]['Q'] = i
    else: 
        emb_dict[k]['Q'] = i
    i+=1
    # QA.append(dataset[k]['Q'])
    # QA.append(dataset[k]['A'][0])
    for f in dataset[k]['txt_negFacts']:
        txt.append(f['fact'])
        if 'txt_negFacts' not in emb_dict[k]:
            emb_dict[k]['txt_negFacts'] = {f['snippet_id'] : i}
        else:
            emb_dict[k]['txt_negFacts'][f['snippet_id']] = i
        i+=1
    for f in dataset[k]['txt_posFacts']:
        txt.append(f['fact'])
        if 'txt_posFacts' not in emb_dict[k]:
            emb_dict[k]['txt_posFacts'] = {f['snippet_id'] : i}
        else:
            emb_dict[k]['txt_posFacts'][f['snippet_id']] = i
        i+=1
    for f in dataset[k]['img_negFacts']:
        txt.append(f['caption'])
        if 'img_negFacts' not in emb_dict[k]:
            emb_dict[k]['img_negFacts'] = {f['image_id'] : i}
        else:
            emb_dict[k]['img_negFacts'][f['image_id']] = i
        i+=1
    for f in dataset[k]['img_posFacts']:
        txt.append(f['caption'])
        if 'img_posFacts' not in emb_dict[k]:
            emb_dict[k]['img_posFacts'] = {f['image_id'] : i}
        else:
            emb_dict[k]['img_posFacts'][f['image_id']] = i
        i+=1
    if (q_indx+1)%batch_size==0:
        embeddings = model.encode(txt)
        print(embeddings.shape)
        for k in keys:
            emb_dict[k]['Q'] = embeddings[emb_dict[k]['Q']]
            if 'txt_negFacts' in emb_dict[k]:
                for id, v in emb_dict[k]['txt_negFacts'].items():
                    emb_dict[k]['txt_negFacts'][id] = embeddings[v]
            if 'txt_posFacts' in emb_dict[k]:
                for id, v in emb_dict[k]['txt_posFacts'].items():
                    emb_dict[k]['txt_posFacts'][id] = embeddings[v]
            if 'img_posFacts' in emb_dict[k]:
                for id, v in emb_dict[k]['img_posFacts'].items():
                    emb_dict[k]['img_posFacts'][id] = embeddings[v]
            if 'img_negFacts' in emb_dict[k]:
                for id, v in emb_dict[k]['img_negFacts'].items():
                    emb_dict[k]['img_negFacts'][id] = embeddings[v]
        with open(save_path,"wb") as fp:
            pickle.dump(emb_dict,fp)
            print("done writing {} questions".format(q_indx))
        txt = []
        i=0
        keys = []
# print(emb_dict)
#remainder samples
embeddings = model.encode(txt)
print(embeddings.shape)
for k in keys:
    emb_dict[k]['Q'] = embeddings[emb_dict[k]['Q']]
    if 'txt_negFacts' in emb_dict[k]:
        for id, v in emb_dict[k]['txt_negFacts'].items():
            emb_dict[k]['txt_negFacts'][id] = embeddings[v]
    if 'txt_posFacts' in emb_dict[k]:
        for id, v in emb_dict[k]['txt_posFacts'].items():
            emb_dict[k]['txt_posFacts'][id] = embeddings[v]
    if 'img_posFacts' in emb_dict[k]:
        for id, v in emb_dict[k]['img_posFacts'].items():
            emb_dict[k]['img_posFacts'][id] = embeddings[v]
    if 'img_negFacts' in emb_dict[k]:
        for id, v in emb_dict[k]['img_negFacts'].items():
            emb_dict[k]['img_negFacts'][id] = embeddings[v]
with open(save_path,"wb") as fp:
    pickle.dump(emb_dict,fp)