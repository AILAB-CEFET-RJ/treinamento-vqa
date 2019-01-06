#################################################################
#             Script para gerar arquivo de triplas              #
#################################################################

import os
import itertools
import numpy as np
import pandas as pd
from random import shuffle

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")

def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return pname

def criar_triplas(lista_imagens):    
    data = pd.read_csv(lista_imagens, sep=",", header=0, names=["image_id","filename","category_id"])
    img_groups = {}
    
    for index, row in data.iterrows():
        pid = row["filename"]
        gid = row["category_id"]
        
        if gid in img_groups:
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    
    pos_triples, neg_triples = [], []
    #A triplas positivas são a combinação de imagens com a mesma categoria
    for key in img_groups.keys():
        triples = [(x[0], x[1], 1) 
                 for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)
    # é necessário o mesmo número de exemplos negativos
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    return pos_triples

lista_imagens = os.path.join(DATA_DIR, 'train_2014.csv')
triplas_imagens = criar_triplas(lista_imagens)

print("# triplas train:", len(triplas_imagens))
[x for x in triplas_imagens[0:5]]

df = pd.DataFrame(triplas_imagens, columns=['left','right','similar'])
df.to_csv(os.path.join(DATA_DIR, "triples_train.csv"))
print("Salvo")

lista_imagens = os.path.join(DATA_DIR, 'val_2014.csv')
triplas_imagens = criar_triplas(lista_imagens)
print("# triplas validation:", len(triplas_imagens))
[x for x in triplas_imagens[0:5]]

df = pd.DataFrame(triplas_imagens, columns=['left','right','similar'])
df.to_csv(os.path.join(DATA_DIR, "triples_val.csv"))
print("Salvo")

print("Finzalizado com sucesso !!!")