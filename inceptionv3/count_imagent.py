import sys, os, time, logging
import pandas as pd
import numpy as np


def load_synset_list(path):
    df = pd.read_csv(path, encoding="utf-8", sep=' ', usecols=[0])
    return df.values

DATA_DIR = os.environ["DATA_DIR"]

data = load_synset_list(os.path.join(DATA_DIR, "LOC_synset_mapping.txt"))

print(len(data[0:50]))