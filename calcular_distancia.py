import os
import sys
import numpy as np
import pandas as pd
from random import shuffle
from utils import calc, dados
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")

def preprocessar_dados(vec_dict, triplas, train_size=0.7):
    data = []
    i, tam = 1, len(image_triples)
    for image_triple in triplas:
        X1 = vec_dict[image_triple[0]]
        X2 = vec_dict[image_triple[1]]
        distance = calc.euclidian_distance([X1, X2])       
        data.append([distance[0], image_triple[2]])

        if(i % 10000 == 0):            
            print("Processado", i , "de", tam)
        i = i + 1
     
    return np.array(data)

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict

print("Carregando vetores")
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")
vec_dict = carregar_vetores(VECTOR_FILE)

print("Carregando triplas")
TRIPLES_FILES = os.path.join("data/", "triples_train.csv")
image_triples = dados.carregar_triplas(TRIPLES_FILES)

print("Calculando distancia")
data = preprocessar_dados(vec_dict, image_triples)

df = pd.DataFrame(data, columns=['distance', 'similar'])
print(df.head(25))
df.to_csv(os.path.join("data", "distances.csv"))

print("Salvo")
print("Finalizado")